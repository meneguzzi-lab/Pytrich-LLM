import argparse
import os
import re
import csv
import glob

class LogParser:
    def __init__(self):
        self.results = []
        self.total_files_processed = 0
        self.skipped_files = 0
    
    def _detect_log_format(self, content):
        """Detect whether the log is Pytrich or Panda format."""
        if 'Domain :' in content and 'Problem :' in content:
            return 'pytrich'
        elif 'Reading input from' in content and '.sas' in content:
            return 'panda'
        return 'unknown'

    def parse_log_file(self, file_path):
        """Parse a single log file and extract required fields."""
        data = self._get_empty_data()

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            log_format = self._detect_log_format(content)

            if log_format == 'pytrich':
                data = self._parse_pytrich_log(content)
            elif log_format == 'panda':
                data = self._parse_panda_log(content)
            # else: return empty data for unknown format

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

        return data

    def _get_empty_data(self):
        """Return an empty data dictionary."""
        return {
            'domain_name': None,
            'problem_name': None,
            'experiment_name': None,
            'search_elapsed_time': None,
            'solution_size': None,
            'expanded_nodes': None,
            'nodes_per_second': None,
            'heuristic_name': None,
            'heuristic_elapsed_time': None,
            'initial_heuristic_value': None,
            'total_landmarks': None,
            'disj_landmarks': None,
            'abtask_landmarks': None,
            'operator_landmarks': None,
            'method_landmarks': None,
            'fact_landmarks': None
        }

    def _parse_pytrich_log(self, content):
        """Parse Pytrich log format."""
        data = self._get_empty_data()

        for raw_line in content.split('\n'):
            line = raw_line.strip()

            # Extract values using simple string matching
            if line.startswith('Experiment Name :'):
                data['experiment_name'] = line.split(':', 1)[1].strip()

            elif line.startswith('Domain :'):
                data['domain_name'] = line.split(':', 1)[1].strip()

            elif line.startswith('Problem :'):
                problem = line.split(':', 1)[1].strip()
                # Remove the problem-file extension if present (.pddl in Lamps)
                if problem.endswith(('.hddl', '.pddl')):
                    problem = problem[:-5]
                data['problem_name'] = problem

            elif line.startswith('Search Time (seconds) :'):
                data['search_elapsed_time'] = self._extract_number(line)

            elif line.startswith('Solution Size :'):
                data['solution_size'] = self._extract_number(line)

            elif line.startswith('Nodes Expanded :'):
                data['expanded_nodes'] = self._extract_number(line)

            elif line.startswith('Nodes per Second :'):
                data['nodes_per_second'] = self._extract_number(line)

            elif 'Heuristic:' in line:
                # Extract from "  Heuristic: robot_heuristic_0, Params: {}"
                match = re.search(r'Heuristic:\s*([^,\s]+)', line)
                if match:
                    heuristic = match.group(1).strip()
                    # Extract just the heuristic number part
                    heuristic_match = re.search(r'heuristic_(\d+)', heuristic)
                    if heuristic_match:
                        data['heuristic_name'] = f"heuristic_{heuristic_match.group(1)}"
                    else:
                        # If no underscore pattern, use the name as-is (e.g., TDG, Novelty)
                        data['heuristic_name'] = heuristic
                        # Only derive a name when the wrapper did not supply one.
                        # `Experiment Name :` comes from -en and is authoritative. For
                        # ablation arms it is the ONLY place the arm appears: heuristics
                        # loaded via --heuristic-file log their CLASS name, which is
                        # identical across arms, so overwriting here collapsed all five
                        # arms onto one experiment_name and silently mixed them.
                        if not data['experiment_name']:
                            data['experiment_name'] = f"Pytrich-{heuristic}"

            elif line.startswith('\tName:'):
                # Extract from "	Name: robot_heuristic_0"
                heuristic = line.split(':', 1)[1].strip()
                heuristic_match = re.search(r'heuristic_(\d+)', heuristic)
                if heuristic_match:
                    data['heuristic_name'] = f"heuristic_{heuristic_match.group(1)}"

            elif raw_line.startswith('Heuristic Elapsed Time :'):
                # Measured by the search around heuristic.initialize(); unindented, so the
                # tab-indented value some heuristics self-report (e.g. LMCOUNT) is not picked up
                data['heuristic_elapsed_time'] = self._extract_number(line)

            elif line.startswith('\tPreprocessing time:'):
                # Extract from "	Preprocessing time: 0.00 s"
                data['heuristic_elapsed_time'] = self._extract_number(line)

            elif line.startswith('Initial heuristic value (heuristic.py)'):
                data['initial_heuristic_value'] = self._extract_number(line)

        return data

    def _parse_panda_log(self, content):
        """Parse Panda planner log format."""
        data = self._get_empty_data()
        data['experiment_name'] = 'PANDA'  # default fallback

        # Extract experiment_name from wrapper: "Engine: Panda_GBFS"
        engine_match = re.search(r'^Engine:\s*(.+)$', content, re.MULTILINE)
        if engine_match:
            data['experiment_name'] = engine_match.group(1).strip()

        # Extract domain_name and problem_name from .sas file path
        # Pattern: "Reading input from .../Domain-Problem.sas."
        sas_match = re.search(r'Reading input from [^\s]*?([^/]+)-([^-/]+)\.sas', content)
        if sas_match:
            data['domain_name'] = sas_match.group(1)
            data['problem_name'] = sas_match.group(2)

        # Extract search_elapsed_time: "- Search time 0.001 seconds"
        time_match = re.search(r'-\s*Search time\s+([\d.]+)\s+seconds', content)
        if time_match:
            data['search_elapsed_time'] = float(time_match.group(1))

        # Extract solution_size from the plan block. "Found solution of length N" counts
        # every node of the decomposition tree: primitive actions, the grounder's
        # pseudo-actions, and one entry per method applied. The paper's plan length is
        # the number of primitive actions, which is what Pytrich reports, so count the
        # primitive lines of the block (between "==>" and "root") and skip the
        # pseudo-actions. These all start with "__" (an HDDL name must start with a
        # letter): __method_precondition_* and __noop, which the grounder adds for an
        # empty method (one per Robot plan in S1). Fall back to the reported length only
        # when the log has no plan block (then it cannot be compared with Pytrich anyway).
        plan_block = re.search(r'^==>\s*$(.*?)^root\b', content, re.MULTILINE | re.DOTALL)
        if plan_block:
            primitives = [
                line for line in plan_block.group(1).splitlines()
                if re.match(r'\s*\d+\s+\S', line)
                and not re.match(r'\s*\d+\s+__', line)
            ]
            data['solution_size'] = len(primitives)
        else:
            solution_match = re.search(r'-\s*Found solution of length\s+(\d+)', content)
            if solution_match:
                data['solution_size'] = int(solution_match.group(1))

        # Extract expanded_nodes: "- Generated 179 search nodes"
        nodes_match = re.search(r'-\s*Generated\s+(\d+)\s+search nodes', content)
        if nodes_match:
            data['expanded_nodes'] = int(nodes_match.group(1))

        # Extract nodes_per_second: "- Generated 128000 nodes per second"
        nps_match = re.search(r'-\s*Generated\s+(-?\d+)\s+nodes per second', content)
        if nps_match:
            nps_value = int(nps_match.group(1))
            # Panda sometimes outputs negative values for overflow, skip those
            if nps_value > 0:
                data['nodes_per_second'] = nps_value

        return data
    
    def _extract_number(self, line):
        """Extract numeric value from line."""
        value = line.split(':', 1)[1].strip()
        # Find first number (integer or float)
        match = re.search(r'[\d.]+', value)
        if match:
            try:
                num = float(match.group())
                return int(num) if num.is_integer() else num
            except ValueError:
                pass
        return None
    
    def parse_directory(self, root_path, file_pattern="*.txt,*.out,*.log"):
        """Parse all matching files recursively. Supports comma-separated patterns."""
        patterns = [p.strip() for p in file_pattern.split(',')]
        print(f"Searching for {patterns} files in: {root_path}")

        log_files = []
        for pat in patterns:
            pattern = os.path.join(root_path, "**", pat)
            log_files.extend(glob.glob(pattern, recursive=True))

        # Remove duplicates and sort
        log_files = sorted(set(log_files))

        if not log_files:
            print(f"No files found matching patterns: {patterns}")
            return

        print(f"Found {len(log_files)} files to process\n")
        
        for file_path in log_files:
            print(f"Processing: {file_path}")
            data = self.parse_log_file(file_path)
            self.total_files_processed += 1

            # Require all essential fields for a valid solved result
            if (data['domain_name'] and data['problem_name'] and
                data['expanded_nodes'] is not None and data['solution_size'] is not None):
                self.results.append(data)
                print(f"  ✓ {data['domain_name']} - {data['problem_name']} - {data['heuristic_name']}")
            else:
                # Provide detailed reason for exclusion
                self.skipped_files += 1
                missing = []
                if not data['domain_name']:
                    missing.append('domain')
                if not data['problem_name']:
                    missing.append('problem')
                if data['expanded_nodes'] is None:
                    missing.append('expanded_nodes')
                if data['solution_size'] is None:
                    missing.append('solution_size')
                print(f"  ✗ Missing: {', '.join(missing)}")
    
    def save_to_csv(self, output_file):
        """Save results to CSV with exact format specified."""
        if not self.results:
            print("No data to save")
            return
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Define headers in exact order
        headers = [
            'domain_name', 'problem_name', 'experiment_name', 'search_elapsed_time',
            'solution_size', 'expanded_nodes', 'nodes_per_second', 'heuristic_name',
            'heuristic_elapsed_time', 'initial_heuristic_value', 'total_landmarks', 'disj_landmarks',
            'abtask_landmarks', 'operator_landmarks', 'method_landmarks', 'fact_landmarks'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for data in self.results:
                row = [data.get(header, '') for header in headers]
                writer.writerow(row)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Total records: {len(self.results)}")
    
    def print_summary(self):
        """Print summary of results."""
        if not self.results:
            print("No results to summarize")
            return

        domains = set(r['domain_name'] for r in self.results if r['domain_name'])
        heuristics = set(r['heuristic_name'] for r in self.results if r['heuristic_name'])
        experiments = set(r['experiment_name'] for r in self.results if r['experiment_name'])

        print(f"\n--- Summary ---")
        print(f"Files processed: {self.total_files_processed}")
        print(f"Valid records: {len(self.results)}")
        print(f"Skipped (unsolved/missing data): {self.skipped_files}")
        print(f"Domains: {sorted(domains)}")
        print(f"Experiments: {sorted(experiments)}")
        print(f"Heuristics: {sorted(heuristics)}")

def main():
    parser = argparse.ArgumentParser(description="Parse experiment log files to CSV")
    parser.add_argument("-i", "--input", required=True,
                       help="Root directory containing log files")
    parser.add_argument("-o", "--output", required=True,
                       help="Output CSV file path")
    parser.add_argument("--pattern", default="*.txt,*.out,*.log",
                       help="File pattern(s), comma-separated (default: *.txt,*.out,*.log)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        return
    
    # Parse files
    log_parser = LogParser()
    log_parser.parse_directory(args.input, args.pattern)
    
    # Save results
    if log_parser.results:
        log_parser.save_to_csv(args.output)
        log_parser.print_summary()
    else:
        print("No valid log files found to process")

if __name__ == "__main__":
    main()