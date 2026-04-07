import docker
import time
from pathlib import Path
import os
import tarfile
import io
import threading
import sys
import socket
import json

# --- Constants ---
DEFAULT_IMAGE = 'multilang-autograder:latest' # Must match your built image name
# Fallback if custom image not found
FALLBACK_IMAGE = 'ubuntu:22.04' 

CONTAINER_WORKING_DIR = '/usr/src/app'
CONTAINER_TEMP_DIR = '/tmp'
RUNNER_SCRIPT_NAME = 'runner.py'
INPUT_FILE_NAME = 'input.txt'
EXECUTION_TIMEOUT_SECONDS = 5.0

class DynamicAnalyzer:
    def __init__(self):
        self.client = None
        try:
            self.client = docker.from_env()
            self.client.ping()
            print("[DYNAMIC] Docker client initialized.")
        except Exception as e:
            print(f"[DYNAMIC] Docker init error: {e}")

    def _create_tar_from_string(self, content_str: str, filename: str) -> io.BytesIO:
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode='w:') as tar:
            data = content_str.encode('utf-8')
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
        tar_stream.seek(0)
        return tar_stream

    def _get_execution_env(self, language: str, code_path: Path, input_path: str) -> dict:
        """
        Returns compilation and execution commands for the given language.
        """
        filename = code_path.name
        stem = code_path.stem
        
        env = {
            "compile": None,
            "run": None,
            "is_interpreted": False
        }

        if language == 'python':
            env["run"] = ["python3", "-u", filename]
            env["is_interpreted"] = True
        elif language == 'c':
            env["compile"] = ["gcc", filename, "-o", "/tmp/main", "-lm"]
            env["run"] = ["/tmp/main"]
        elif language == 'cpp':
            env["compile"] = ["g++", filename, "-o", "/tmp/main"]
            env["run"] = ["/tmp/main"]
        elif language == 'java':
            # Java is tricky from a separate dir, so we copy all .java files to /tmp first
            env["compile"] = ["/bin/sh", "-c", f"cp *.java /tmp/ && javac /tmp/{filename}"]
            env["run"] = ["java", "-cp", "/tmp", stem]
        elif language == 'javascript':
            env["run"] = ["node", filename]
            env["is_interpreted"] = True
        
        return env

    def _run_test_case_in_container(self, code_path: Path, language: str, input_data: str, mode: dict) -> tuple[int | None, str, str, str | None]:
        """
        Runs a test case. Returns (exit_code, stdout, stderr, compile_error).
        """
        container = None
        compile_error = None
        try:
            # Check if image exists, else fallback
            try:
                self.client.images.get(DEFAULT_IMAGE)
                image_to_use = DEFAULT_IMAGE
            except docker.errors.ImageNotFound:
                print(f"[DYNAMIC] WARNING: Image {DEFAULT_IMAGE} not found. Using {FALLBACK_IMAGE}.")
                image_to_use = FALLBACK_IMAGE

            volume_mount = {
                str(code_path.parent.resolve()): {
                    'bind': CONTAINER_WORKING_DIR, 'mode': 'ro'
                }
            }
            
            container = self.client.containers.run(
                image_to_use,
                command=['/bin/sh', '-c', 'sleep infinity'],
                detach=True, volumes=volume_mount,
                working_dir=CONTAINER_WORKING_DIR, mem_limit='512m'
            )

            # 1. Prepare Input File
            input_target = f"{CONTAINER_TEMP_DIR}/{INPUT_FILE_NAME}"
            input_tar = self._create_tar_from_string(input_data, INPUT_FILE_NAME)
            container.put_archive(path=CONTAINER_TEMP_DIR, data=input_tar)

            # 2. Get Commands
            exec_env = self._get_execution_env(language, code_path, input_target)
            
            # 3. Compilation Step (if needed)
            if exec_env["compile"]:
                res = container.exec_run(exec_env["compile"], demux=True)
                if res.exit_code != 0:
                    compile_error = res.output[1].decode('utf-8', errors='ignore') if res.output[1] else "Compilation failed."
                    return None, "", "", compile_error

            # 4. Execution Step
            # For compiled languages, we use shell to redirect input.txt
            if not exec_env["is_interpreted"] or language == 'javascript':
                run_cmd_str = " ".join(exec_env["run"])
                exec_command = ["/bin/sh", "-c", f"{run_cmd_str} < {input_target}"]
            else:
                # Python specific runner if needed, or just straight python
                exec_command = exec_env["run"] + ["<", input_target] # Shell redirection still better
                exec_command = ["/bin/sh", "-c", f"{' '.join(exec_env['run'])} < {input_target}"]

            exit_code_ref = [None]
            output_bytes_ref = [None]
            error_ref = [None]

            def exec_target():
                try:
                    ec, output = container.exec_run(exec_command, demux=True)
                    exit_code_ref[0] = ec
                    output_bytes_ref[0] = output if output else (b'', b'')
                except Exception as e:
                    error_ref[0] = e

            thread = threading.Thread(target=exec_target)
            thread.start()
            thread.join(EXECUTION_TIMEOUT_SECONDS)

            if thread.is_alive():
                try: container.stop(timeout=1)
                except: pass
                raise TimeoutError("Code execution timed out.")

            if error_ref[0]: raise error_ref[0]

            exit_code = exit_code_ref[0]
            stdout_bytes, stderr_bytes = output_bytes_ref[0] if output_bytes_ref[0] else (b'', b'')

            stdout_decoded = stdout_bytes.decode('utf-8', errors='ignore').strip() if stdout_bytes else ''
            stderr_decoded = stderr_bytes.decode('utf-8', errors='ignore').strip() if stderr_bytes else ''

            return exit_code, stdout_decoded, stderr_decoded, None

        finally:
            if container:
                try: container.remove(force=True)
                except Exception as e:
                    print(f"[CLEANUP ERROR] {e}")

    def analyze(self, submission: dict) -> dict:
        student_id = submission.get("student_id")
        print(f"\n[\U0001F50D] Analyzing submission for: {student_id}")

        if not self.client:
            print("[❌] Docker client unavailable.")
            submission['analysis']['dynamic'] = [{"name": "all_tests", "status": "skipped", "error": "Docker unavailable"}]
            return submission

        code_path = Path(submission['code_path'])
        language = submission.get('language', 'python')
        config = submission['config']
        mode_config = config.get('execution_mode', {'type': 'program'})

        all_test_cases = config.get("test_cases", [])
        edge_cases = config.get("edge_cases", [])
        
        # Combine tests but keep track of which are edge cases
        test_queue = [(t, False) for t in all_test_cases] + [(t, True) for t in edge_cases]

        results = []
        for test, is_edge in test_queue:
            name = test.get("name", "test")
            if is_edge: name = f"[EDGE] {name}"
            
            input_data_raw = test.get("input", "")
            expected = test.get("expected_output", "")
            expected_str = str(expected).strip() if isinstance(expected, (str, int, float, list, dict)) else ""

            try:
                if isinstance(input_data_raw, (list, dict)):
                    input_str = json.dumps(input_data_raw)
                else:
                    input_str = str(input_data_raw)
            except Exception as e:
                print(f"[ERROR] Invalid test input for '{name}': {e}")
                input_str = str(input_data_raw)

            print(f"\n[TEST] Running {name}...")

            try:
                exit_code, stdout_log, stderr_log, compile_error = self._run_test_case_in_container(code_path, language, input_str, mode_config)

                if compile_error:
                    print(f"[RESULT] {name} → 🛑 COMPILATION ERROR")
                    results.append({"name": name, "status": "compilation_error", "error": compile_error, "is_edge": is_edge})
                    continue

                print(f"    [DEBUG] Exit Code: {exit_code}")
                print(f"    [DEBUG] STDOUT: {repr(stdout_log)}")
                if stderr_log:
                    print(f"    [DEBUG] STDERR: {repr(stderr_log)}")

                status = ""
                error = ""

                if exit_code is None:
                    status = "system_error"
                    error = "No exit code returned."
                elif exit_code != 0:
                    status = "runtime_error"
                    error = stderr_log if stderr_log else "Runtime error."
                elif stdout_log.strip() == expected_str:
                    status = "pass"
                else:
                    status = "fail"

                result_dict = {"name": name, "status": status, "is_edge": is_edge}
                if status == "pass":
                    print(f"[RESULT] {name} → ✅ PASS")
                elif status == "fail":
                    print(f"[RESULT] {name} → ❌ FAIL")
                    result_dict.update({"expected": expected_str, "actual": stdout_log, "stderr_on_fail": stderr_log})
                elif status == "runtime_error":
                    print(f"[RESULT] {name} → 💥 RUNTIME ERROR")
                    result_dict.update({"error": error})
                elif status == "system_error":
                    print(f"[RESULT] {name} → 🚨 SYSTEM ERROR")
                    result_dict.update({"error": error})
                results.append(result_dict)

            except TimeoutError as e:
                print(f"[RESULT] {name} → ⏰ TIMEOUT")
                results.append({"name": name, "status": "timeout", "error": str(e), "is_edge": is_edge})
            except Exception as e:
                print(f"[RESULT] {name} → 🛑 UNEXPECTED ERROR: {str(e)}")
                results.append({"name": name, "status": "system_error", "error": f"Unexpected exception: {e}", "is_edge": is_edge})

        submission['analysis']['dynamic'] = results
        print(f"\n[✅] Completed analysis for {student_id}")
        return submission
