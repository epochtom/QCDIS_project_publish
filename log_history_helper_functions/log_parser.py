# -*- coding: utf-8 -*-
"""
Log Parser
Parses debug.log to extract execution information and errors.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class LogParser:
    """Parses execution logs to extract errors and execution information"""
    
    def __init__(self, log_file: str = "debug.log"):
        """
        Initialize log parser
        
        Args:
            log_file: Path to debug log file
        """
        self.log_file = Path(log_file)
    
    def parse_log(self) -> List[Dict]:
        """
        Parse the entire log file and extract execution blocks
        
        Returns:
            List of execution blocks with metadata
        """
        if not self.log_file.exists():
            return []
        
        execution_blocks = []
        current_block = None
        
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    
                    # Check for execution start
                    if "Execution Started" in line or "Docker Execution Started" in line or "MATLAB Execution Started" in line:
                        if current_block:
                            execution_blocks.append(current_block)
                        
                        # Extract script name and timestamp
                        script_name = self._extract_script_name(line)
                        timestamp = self._extract_timestamp(line)
                        
                        current_block = {
                            "script_name": script_name,
                            "timestamp": timestamp,
                            "type": "docker" if "Docker" in line else "matlab" if "MATLAB" in line else "unknown",
                            "lines": [line],
                            "error_log": None,
                            "success": None
                        }
                    elif current_block:
                        current_block["lines"].append(line)
                        
                        # Check for execution completion
                        if "Execution Completed Successfully" in line:
                            current_block["success"] = True
                            execution_blocks.append(current_block)
                            current_block = None
                        elif "Execution Failed" in line or "failed with exit code" in line.lower():
                            current_block["success"] = False
                            # Extract error information
                            current_block["error_log"] = self._extract_error_from_block(current_block["lines"])
                            execution_blocks.append(current_block)
                            current_block = None
                        elif "=" * 80 in line and current_block and len(current_block["lines"]) > 5:
                            # End of execution block (separator line)
                            if current_block["success"] is None:
                                # Assume failure if we hit separator without success marker
                                current_block["success"] = False
                                current_block["error_log"] = self._extract_error_from_block(current_block["lines"])
                            execution_blocks.append(current_block)
                            current_block = None
            
            # Add final block if exists
            if current_block:
                execution_blocks.append(current_block)
        
        except Exception as e:
            print(f"Error parsing log file: {e}")
        
        return execution_blocks
    
    def _extract_script_name(self, line: str) -> Optional[str]:
        """Extract script name from log line"""
        # Look for "Script: <name>" pattern
        match = re.search(r'Script:\s*([^\s]+)', line)
        if match:
            return match.group(1)
        
        # Look in subsequent lines if available
        return None
    
    def _extract_timestamp(self, line: str) -> Optional[str]:
        """Extract timestamp from log line"""
        # Look for [YYYY-MM-DD HH:MM:SS] pattern
        match = re.search(r'\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]', line)
        if match:
            return match.group(1)
        return None
    
    def _extract_error_from_block(self, lines: List[str]) -> str:
        """Extract error message from execution block"""
        error_lines = []
        in_error = False
        
        for line in lines:
            line_lower = line.lower()
            
            # Error indicators
            if any(keyword in line_lower for keyword in ["error", "exception", "traceback", "failed", "exit code"]):
                in_error = True
            
            if in_error:
                error_lines.append(line)
            
            # Stop at separator or success message
            if "=" * 80 in line or "completed successfully" in line_lower:
                break
        
        # If no explicit error found, look for exit code
        if not error_lines:
            for line in lines:
                if "exit code" in line.lower() and "0" not in re.findall(r'exit code[:\s]+(\d+)', line.lower()):
                    error_lines.append(line)
        
        # Return last 20 lines of error (to avoid too much text)
        return "\n".join(error_lines[-20:]) if error_lines else "Execution failed (see debug.log for details)"
    
    def get_errors_for_script(self, script_name: str) -> List[Dict]:
        """Get all error logs for a specific script"""
        blocks = self.parse_log()
        errors = []
        
        for block in blocks:
            if block["script_name"] == script_name and block["success"] is False:
                errors.append({
                    "timestamp": block["timestamp"],
                    "error_log": block["error_log"],
                    "type": block["type"]
                })
        
        return errors
    
    def get_latest_execution(self, script_name: str) -> Optional[Dict]:
        """Get the most recent execution for a script"""
        blocks = self.parse_log()
        
        script_blocks = [b for b in blocks if b["script_name"] == script_name]
        if not script_blocks:
            return None
        
        # Sort by timestamp (most recent first)
        script_blocks.sort(key=lambda x: x["timestamp"] or "", reverse=True)
        return script_blocks[0]

