# -*- coding: utf-8 -*-
"""
Diff Calculator
Calculates differences between script versions.
"""

import difflib
from typing import Optional, List, Dict


class DiffCalculator:
    """Calculates and formats differences between script versions"""
    
    @staticmethod
    def calculate_diff(content1: str, content2: str, 
                      from_version: int, to_version: int) -> str:
        """
        Calculate unified diff between two script versions
        
        Args:
            content1: Content of first version (older)
            content2: Content of second version (newer)
            from_version: Version number of first content
            to_version: Version number of second content
        
        Returns:
            Formatted diff string
        """
        lines1 = content1.splitlines(keepends=True)
        lines2 = content2.splitlines(keepends=True)
        
        # Generate unified diff
        diff = difflib.unified_diff(
            lines1,
            lines2,
            fromfile=f"version_{from_version}",
            tofile=f"version_{to_version}",
            lineterm='',
            n=3  # Context lines
        )
        
        return ''.join(diff)
    
    @staticmethod
    def calculate_diff_summary(content1: str, content2: str) -> Dict:
        """
        Calculate summary statistics of differences
        
        Returns:
            Dictionary with added, removed, modified line counts
        """
        lines1 = content1.splitlines()
        lines2 = content2.splitlines()
        
        diff = difflib.SequenceMatcher(None, lines1, lines2)
        
        # Calculate sizes from indices
        # For 'insert': size is j2 - j1 (new lines added)
        # For 'delete': size is i2 - i1 (old lines removed)
        # For 'replace': count both old and new lines as modified
        added = sum(j2 - j1 for tag, i1, i2, j1, j2 in diff.get_opcodes() if tag == 'insert')
        removed = sum(i2 - i1 for tag, i1, i2, j1, j2 in diff.get_opcodes() if tag == 'delete')
        modified = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in diff.get_opcodes() if tag == 'replace')
        
        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "similarity": diff.ratio()
        }
    
    @staticmethod
    def format_diff_for_display(diff: str, max_lines: int = 50) -> str:
        """
        Format diff for table display (truncated)
        
        Args:
            diff: Full diff string
            max_lines: Maximum lines to show
        
        Returns:
            Truncated diff string
        """
        lines = diff.split('\n')
        
        if len(lines) <= max_lines:
            return diff
        
        # Show first and last parts
        first_part = '\n'.join(lines[:max_lines // 2])
        last_part = '\n'.join(lines[-max_lines // 2:])
        
        return f"{first_part}\n... ({len(lines) - max_lines} more lines) ...\n{last_part}"
    
    @staticmethod
    def get_diff_preview(content1: str, content2: str) -> str:
        """
        Get a short preview of changes (for table display)
        
        Returns:
            Short summary of changes
        """
        summary = DiffCalculator.calculate_diff_summary(content1, content2)
        
        changes = []
        if summary["added"] > 0:
            changes.append(f"+{summary['added']} lines")
        if summary["removed"] > 0:
            changes.append(f"-{summary['removed']} lines")
        if summary["modified"] > 0:
            changes.append(f"~{summary['modified']} modified")
        
        if changes:
            return ", ".join(changes)
        else:
            return "No changes"

