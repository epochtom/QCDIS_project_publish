# -*- coding: utf-8 -*-
"""
Version History Manager
Tracks script versions, modifications, and execution results.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List


class VersionHistoryManager:
    """Manages script version history and metadata"""
    
    def __init__(self, versions_dir: str = "script_versions", metadata_file: str = "version_metadata.json", auto_recover: bool = True):
        """
        Initialize version history manager
        
        Args:
            versions_dir: Directory to store script versions
            metadata_file: JSON file to store version metadata
            auto_recover: If True, automatically recover missing versions on initialization
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(exist_ok=True)
        self.metadata_file = self.versions_dir / metadata_file
        self.metadata = self._load_metadata()
        
        # Auto-recover missing versions for all scripts
        if auto_recover:
            self._auto_recover_all_scripts()
    
    def _auto_recover_all_scripts(self):
        """Automatically recover missing versions for all scripts found in metadata or version files"""
        # Get all scripts from metadata
        scripts_in_metadata = set(self.metadata.keys())
        
        # Also scan for version files to find scripts not in metadata
        all_scripts = set(scripts_in_metadata)
        for version_file in self.versions_dir.glob("*_v*.py"):
            # Extract script name from filename (e.g., "Python_PQK_NN.py_v2.py" -> "Python_PQK_NN.py")
            filename = version_file.name
            # Find the last occurrence of "_v" and extract everything before it
            if "_v" in filename:
                script_name = filename[:filename.rfind("_v")] + ".py"
                all_scripts.add(script_name)
        
        for version_file in self.versions_dir.glob("*_v*.m"):
            filename = version_file.name
            if "_v" in filename:
                script_name = filename[:filename.rfind("_v")] + ".m"
                all_scripts.add(script_name)
        
        # Recover missing versions for each script
        for script_name in all_scripts:
            recovered = self.recover_missing_versions(script_name)
            if recovered > 0:
                print(f"Auto-recovered {recovered} missing version(s) for {script_name}")
        
        # Migrate existing versions to include user_prompt field if missing
        self._migrate_user_prompt_field()
    
    def _load_metadata(self) -> Dict:
        """Load version metadata from JSON file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    # Migrate old versions to include user_prompt field
                    self._migrate_user_prompt_field_in_metadata(metadata)
                    return metadata
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return {}
        return {}
    
    def _migrate_user_prompt_field_in_metadata(self, metadata: Dict):
        """Add user_prompt field to versions that don't have it (migration for old data)"""
        updated = False
        for script_name, script_data in metadata.items():
            if "versions" in script_data:
                for version_entry in script_data["versions"]:
                    if "user_prompt" not in version_entry:
                        version_entry["user_prompt"] = None
                        updated = True
        if updated:
            # Save migrated metadata
            try:
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                print("DEBUG: Migrated existing versions to include user_prompt field")
            except Exception as e:
                print(f"Warning: Could not save migrated metadata: {e}")
    
    def _migrate_user_prompt_field(self):
        """Add user_prompt field to all versions that don't have it"""
        updated = False
        for script_name, script_data in self.metadata.items():
            if "versions" in script_data:
                for version_entry in script_data["versions"]:
                    if "user_prompt" not in version_entry:
                        version_entry["user_prompt"] = None
                        updated = True
        if updated:
            self._save_metadata()
            print("DEBUG: Migrated existing versions to include user_prompt field")
    
    def _save_metadata(self):
        """Save version metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def update_user_prompt(self, script_name: str, version: int, user_prompt: str) -> bool:
        """
        Dedicated method to update user_prompt for a specific version.
        This method ensures the prompt is immediately saved to disk and protected.
        
        Args:
            script_name: Name of the script
            version: Version number
            user_prompt: User prompt to save
            
        Returns:
            True if successful, False otherwise
        """
        if script_name not in self.metadata:
            self.metadata[script_name] = {"versions": []}
        
        versions = self.metadata[script_name]["versions"]
        version_entry = next((v for v in versions if v["version"] == version), None)
        
        if not version_entry:
            print(f"Warning: Version {version} not found for {script_name}, cannot update prompt")
            return False
        
        # Update the prompt
        version_entry["user_prompt"] = user_prompt
        version_entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Immediately save to disk
        self._save_metadata()
        
        # Verify it was saved
        disk_metadata = self._load_metadata()
        if script_name in disk_metadata:
            disk_version = next((v for v in disk_metadata[script_name]["versions"] if v["version"] == version), None)
            if disk_version and disk_version.get("user_prompt") == user_prompt:
                print(f"DEBUG: User prompt saved successfully for version {version}: {user_prompt[:50]}...")
                return True
            else:
                print(f"ERROR: User prompt verification failed for version {version}")
                return False
        
        return True
    
    def save_version(self, script_name: str, script_content: str, source: str = "manual", 
                    version: Optional[int] = None, user_prompt: Optional[str] = None) -> int:
        """
        Save a new version of a script
        
        Args:
            script_name: Name of the script file
            script_content: Content of the script
            source: Source of modification ("manual", "llm_modification", "initial_copy", "pre_execution")
            version: Optional version number (auto-increments if None)
            user_prompt: Optional user prompt/input that led to this version (for LLM modifications)
        
        Returns:
            Version number that was saved
        """
        # STEP 1: AUTO-RECOVERY - Ensure all version files are in metadata before saving
        # This prevents losing versions when saving new ones
        # CRITICAL: Only do auto-recovery if we're not explicitly saving with a prompt for a specific version
        # This prevents overwriting prompts when saving LLM modifications
        if not (version is not None and user_prompt and source == "llm_modification"):
            self._ensure_all_file_versions_in_metadata(script_name)
        else:
            # For LLM modifications with explicit version and prompt, do a more careful check
            # Only recover versions that are NOT the one we're about to save
            self._ensure_all_file_versions_in_metadata_selective(script_name, exclude_version=version)
        
        if script_name not in self.metadata:
            self.metadata[script_name] = {"versions": []}
        
        versions = self.metadata[script_name]["versions"]
        
        # Determine version number
        if version is None:
            if versions:
                version = max(v["version"] for v in versions) + 1
            else:
                version = 1
        else:
            # Check if version already exists
            existing_version = next((v for v in versions if v["version"] == version), None)
            if existing_version:
                # For "pre_execution", update the existing version instead of creating a new one
                if source == "pre_execution":
                    # CRITICAL: Before merging, scan version files on disk to ensure we don't lose any versions
                    # This is necessary because multiple VersionHistoryManager instances might exist,
                    # and one instance's in-memory state might not reflect all saved versions
                    self._ensure_all_file_versions_in_metadata(script_name)
                    
                    # Reload metadata from disk to get the most up-to-date state
                    # (This includes any versions that were saved by other instances)
                    disk_metadata = self._load_metadata()
                    
                    # Merge: prioritize in-memory versions (most up-to-date), add any missing from disk
                    if script_name in disk_metadata:
                        disk_versions = disk_metadata[script_name].get("versions", [])
                        # Create a map of version numbers to version entries, starting with in-memory (most recent)
                        version_map = {v["version"]: v for v in versions}  # Start with in-memory versions
                        # Add disk versions that aren't in memory (safety check for versions that might have been lost)
                        # CRITICAL: When merging, preserve user_prompt from in-memory if it exists
                        for disk_v in disk_versions:
                            if disk_v["version"] not in version_map:
                                version_map[disk_v["version"]] = disk_v
                                print(f"DEBUG: Found version {disk_v['version']} on disk metadata that wasn't in memory, adding it back")
                            else:
                                # Version exists in both - preserve user_prompt from in-memory if it has one
                                in_mem_v = version_map[disk_v["version"]]
                                if in_mem_v.get("user_prompt") and not disk_v.get("user_prompt"):
                                    # In-memory has prompt but disk doesn't - keep in-memory version
                                    print(f"DEBUG: Preserving user_prompt from in-memory version {disk_v['version']}")
                                elif disk_v.get("user_prompt") and not in_mem_v.get("user_prompt"):
                                    # Disk has prompt but in-memory doesn't - update in-memory
                                    in_mem_v["user_prompt"] = disk_v["user_prompt"]
                                    print(f"DEBUG: Restoring user_prompt from disk for version {disk_v['version']}")
                        
                        # Update the versions list with merged data, sorted by version number
                        versions = [version_map[v] for v in sorted(version_map.keys())]
                        self.metadata[script_name]["versions"] = versions
                        print(f"DEBUG: Merged versions: in-memory had {len([v for v in version_map.values() if v in versions])}, disk had {len(disk_versions)}, merged total: {len(versions)}")
                    else:
                        # Script not in disk metadata, keep in-memory versions
                        print(f"DEBUG: Script {script_name} not found in disk metadata, keeping in-memory versions")
                    
                    # Re-find the existing version after merge
                    existing_version = next((v for v in versions if v["version"] == version), None)
                    
                    if existing_version:
                        # Preserve original source in a separate field for reference
                        if "original_source" not in existing_version:
                            existing_version["original_source"] = existing_version.get("source", "unknown")
                        
                        # Preserve user_prompt if it exists (don't overwrite it)
                        if "user_prompt" not in existing_version:
                            existing_version["user_prompt"] = None
                        
                        # Update existing version with pre_execution info
                        existing_version["source"] = source
                        existing_version["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # Update the file content
                        version_filename = f"{script_name}_v{version}.py" if script_name.endswith('.py') else f"{script_name}_v{version}.m"
                        version_path = self.versions_dir / version_filename
                        try:
                            with open(version_path, 'w', encoding='utf-8') as f:
                                f.write(script_content)
                            existing_version["file_path"] = str(version_path)
                            self._save_metadata()
                            print(f"DEBUG: Updated existing version {version} with pre_execution info (original source: {existing_version.get('original_source', 'unknown')})")
                            print(f"DEBUG: Total versions after update: {[v['version'] for v in versions]}")
                            return version
                        except Exception as e:
                            print(f"Error updating version file: {e}")
                            return version
                else:
                    # For other sources (like llm_modification), check if we should update existing version
                    # If the existing version doesn't have a user_prompt and we're providing one, update it
                    # CRITICAL: Check if user_prompt is None or empty string (both should be updated)
                    existing_prompt = existing_version.get("user_prompt")
                    # Check if prompt is None, empty string, or not set at all
                    prompt_is_empty = (existing_prompt is None or 
                                      existing_prompt == "" or 
                                      (isinstance(existing_prompt, str) and existing_prompt.strip() == ""))
                    
                    should_update_prompt = (source == "llm_modification" and 
                                           user_prompt and 
                                           prompt_is_empty)
                    
                    print(f"DEBUG: save_version - version {version} exists, existing_prompt={existing_prompt}, user_prompt={user_prompt[:50] if user_prompt else None}..., should_update={should_update_prompt}")
                    
                    if should_update_prompt:
                        # Update the existing version with the prompt
                        existing_version["user_prompt"] = user_prompt
                        existing_version["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        existing_version["source"] = source  # Ensure source is set correctly
                        # Update the file content
                        version_filename = f"{script_name}_v{version}.py" if script_name.endswith('.py') else f"{script_name}_v{version}.m"
                        version_path = self.versions_dir / version_filename
                        try:
                            with open(version_path, 'w', encoding='utf-8') as f:
                                f.write(script_content)
                            existing_version["file_path"] = str(version_path)
                            self._save_metadata()
                            print(f"DEBUG: Updated existing version {version} with user_prompt: {user_prompt[:50]}...")
                            # Verify by reading back from the saved metadata structure
                            verify_prompt = existing_version.get("user_prompt")
                            print(f"DEBUG: Verified saved - user_prompt in metadata: {verify_prompt[:50] if verify_prompt else 'None'}...")
                            # Double-check by reloading from disk
                            reloaded_metadata = self._load_metadata()
                            if script_name in reloaded_metadata:
                                reloaded_versions = reloaded_metadata[script_name].get("versions", [])
                                reloaded_version = next((v for v in reloaded_versions if v.get("version") == version), None)
                                if reloaded_version:
                                    reloaded_prompt = reloaded_version.get("user_prompt")
                                    print(f"DEBUG: Double-check from disk - user_prompt: {reloaded_prompt[:50] if reloaded_prompt else 'None'}...")
                                    if reloaded_prompt != user_prompt:
                                        print(f"ERROR: Prompt not saved correctly to disk! Expected: {user_prompt[:50]}..., Got: {reloaded_prompt[:50] if reloaded_prompt else 'None'}...")
                            return version
                        except Exception as e:
                            print(f"Error updating version file: {e}")
                            import traceback
                            traceback.print_exc()
                            return version
                    else:
                        # For other sources, increment to avoid duplicates
                        version = max(v["version"] for v in versions) + 1
        
        # Save script file
        version_filename = f"{script_name}_v{version}.py" if script_name.endswith('.py') else f"{script_name}_v{version}.m"
        version_path = self.versions_dir / version_filename
        
        try:
            with open(version_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
        except Exception as e:
            print(f"Error saving version file: {e}")
            return version
        
        # Create version entry
        version_entry = {
            "version": version,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "execution_result": None,
            "execution_timestamp": None,
            "error_log": None,
            "file_path": str(version_path),
            "user_prompt": user_prompt  # Store user's input prompt for LLM modifications
        }
        
        versions.append(version_entry)
        # Sort versions by version number to keep them in order
        versions.sort(key=lambda x: x["version"])
        self._save_metadata()
        
        # CRITICAL: If user_prompt was provided, use dedicated method to ensure it's saved
        # This protects against recovery functions overwriting it
        if user_prompt:
            # Reload from disk to ensure we have the latest state
            disk_metadata = self._load_metadata()
            if script_name in disk_metadata:
                disk_version = next((v for v in disk_metadata[script_name]["versions"] if v["version"] == version), None)
                if disk_version and disk_version.get("user_prompt") != user_prompt:
                    # Prompt wasn't saved correctly, use dedicated method
                    print(f"DEBUG: Prompt not saved correctly, using update_user_prompt method")
                    self.update_user_prompt(script_name, version, user_prompt)
        
        # Debug output to verify user_prompt is being saved
        prompt_preview = user_prompt[:50] + "..." if user_prompt and len(user_prompt) > 50 else (user_prompt if user_prompt else "None")
        print(f"DEBUG: Saved version {version} with source '{source}', user_prompt: {prompt_preview}, total versions: {[v['version'] for v in versions]}")
        return version
    
    def get_version_content(self, script_name: str, version: int) -> Optional[str]:
        """Get the content of a specific version"""
        version_filename = f"{script_name}_v{version}.py" if script_name.endswith('.py') else f"{script_name}_v{version}.m"
        version_path = self.versions_dir / version_filename
        
        if version_path.exists():
            try:
                with open(version_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading version file: {e}")
        return None
    
    def get_latest_version(self, script_name: str) -> Optional[int]:
        """Get the latest version number for a script"""
        if script_name not in self.metadata:
            return None
        
        versions = self.metadata[script_name]["versions"]
        if not versions:
            return None
        
        return max(v["version"] for v in versions)
    
    def update_execution_result(self, script_name: str, version: int, 
                               execution_result: str, error_log: Optional[str] = None):
        """
        Update execution result for a version
        
        Args:
            script_name: Name of the script
            version: Version number
            execution_result: "success" or "failed"
            error_log: Optional error log message
        """
        if script_name not in self.metadata:
            return
        
        versions = self.metadata[script_name]["versions"]
        for v in versions:
            if v["version"] == version:
                v["execution_result"] = execution_result
                v["execution_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if error_log:
                    v["error_log"] = error_log
                self._save_metadata()
                return
    
    def get_all_versions(self, script_name: str) -> List[Dict]:
        """Get all version entries for a script, sorted by version number"""
        # STEP 3: AUTO-RECOVERY - Ensure all version files are in metadata before returning
        # This ensures the GUI always shows all versions, even if metadata was incomplete
        self._ensure_all_file_versions_in_metadata(script_name)
        
        if script_name not in self.metadata:
            return []
        
        versions = self.metadata[script_name]["versions"]
        return sorted(versions, key=lambda x: x["version"])
    
    def get_version_info(self, script_name: str, version: int) -> Optional[Dict]:
        """Get information about a specific version"""
        if script_name not in self.metadata:
            return None
        
        versions = self.metadata[script_name]["versions"]
        for v in versions:
            if v["version"] == version:
                return v
        return None
    
    def _ensure_all_file_versions_in_metadata_selective(self, script_name: str, exclude_version: Optional[int] = None):
        """
        Ensure all version files on disk are represented in metadata, excluding a specific version.
        This is used when we're about to save a version with a prompt, to avoid overwriting it.
        
        Args:
            script_name: Name of the script to check
            exclude_version: Version number to exclude from recovery (we're about to save this one)
        """
        if script_name not in self.metadata:
            self.metadata[script_name] = {"versions": []}
        
        versions = self.metadata[script_name]["versions"]
        existing_version_numbers = {v["version"] for v in versions}
        
        # Scan for version files matching the script name pattern
        if script_name.endswith('.py'):
            pattern = f"{script_name}_v*.py"
        else:
            pattern = f"{script_name}_v*.m"
        
        missing_versions = []
        for version_file in self.versions_dir.glob(pattern):
            try:
                # Extract version number from filename
                filename = version_file.name
                if script_name.endswith('.py'):
                    version_str = filename.replace(f"{script_name}_v", "").replace(".py", "")
                else:
                    version_str = filename.replace(f"{script_name}_v", "").replace(".m", "")
                
                try:
                    version_num = int(version_str)
                except ValueError:
                    continue
                
                # Skip the version we're about to save
                if exclude_version is not None and version_num == exclude_version:
                    continue
                
                # Check if this version is in metadata
                if version_num not in existing_version_numbers:
                    # Read file modification time as timestamp
                    file_stat = version_file.stat()
                    file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    # Determine source - if it's between executed versions, it's likely llm_modification
                    source = "llm_modification"  # Default assumption
                    if version_num == 1:
                        source = "initial_copy"
                    
                    # Create version entry
                    version_entry = {
                        "version": version_num,
                        "timestamp": file_mtime.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": source,
                        "execution_result": None,
                        "execution_timestamp": None,
                        "error_log": None,
                        "file_path": str(version_file),
                        "user_prompt": None  # Can't recover prompt from file, set to None
                    }
                    
                    versions.append(version_entry)
                    existing_version_numbers.add(version_num)
                    missing_versions.append(version_num)
                    print(f"DEBUG: Found missing version {version_num} in file {filename}, adding to metadata (excluded {exclude_version})")
            except Exception as e:
                print(f"Warning: Error processing {version_file} in _ensure_all_file_versions_in_metadata_selective: {e}")
        
        if missing_versions:
            # Sort versions by version number
            versions.sort(key=lambda x: x["version"])
            # Save immediately to ensure disk metadata is up-to-date
            self._save_metadata()
            print(f"DEBUG: Added {len(missing_versions)} missing version(s) {missing_versions} to metadata (excluded {exclude_version})")
    
    def _ensure_all_file_versions_in_metadata(self, script_name: str):
        """
        Ensure all version files on disk are represented in metadata.
        This is critical when multiple VersionHistoryManager instances exist,
        as one instance's in-memory state might not reflect all saved versions.
        
        CRITICAL: This function preserves existing user_prompt values from disk.
        
        Args:
            script_name: Name of the script to check
        """
        # First, load existing metadata from disk to preserve prompts
        disk_metadata = self._load_metadata()
        if script_name in disk_metadata:
            # Merge disk metadata with in-memory, preserving prompts from disk
            disk_versions = {v["version"]: v for v in disk_metadata[script_name].get("versions", [])}
            if script_name not in self.metadata:
                self.metadata[script_name] = {"versions": []}
            
            # Update in-memory versions with disk prompts if they exist
            for in_mem_v in self.metadata[script_name]["versions"]:
                disk_v = disk_versions.get(in_mem_v["version"])
                if disk_v and disk_v.get("user_prompt"):
                    # Preserve prompt from disk
                    in_mem_v["user_prompt"] = disk_v["user_prompt"]
            
            # Add any disk versions not in memory
            for disk_v in disk_versions.values():
                existing = next((v for v in self.metadata[script_name]["versions"] if v["version"] == disk_v["version"]), None)
                if not existing:
                    self.metadata[script_name]["versions"].append(disk_v.copy())
        
        if script_name not in self.metadata:
            self.metadata[script_name] = {"versions": []}
        
        versions = self.metadata[script_name]["versions"]
        existing_version_numbers = {v["version"] for v in versions}
        
        # Scan for version files matching the script name pattern
        if script_name.endswith('.py'):
            pattern = f"{script_name}_v*.py"
        else:
            pattern = f"{script_name}_v*.m"
        
        missing_versions = []
        for version_file in self.versions_dir.glob(pattern):
            try:
                # Extract version number from filename
                filename = version_file.name
                if script_name.endswith('.py'):
                    version_str = filename.replace(f"{script_name}_v", "").replace(".py", "")
                else:
                    version_str = filename.replace(f"{script_name}_v", "").replace(".m", "")
                
                try:
                    version_num = int(version_str)
                except ValueError:
                    continue
                
                # Check if this version is in metadata
                if version_num not in existing_version_numbers:
                    # Read file modification time as timestamp
                    file_stat = version_file.stat()
                    file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    # Determine source - if it's between executed versions, it's likely llm_modification
                    source = "llm_modification"  # Default assumption
                    if version_num == 1:
                        source = "initial_copy"
                    
                    # Create version entry
                    version_entry = {
                        "version": version_num,
                        "timestamp": file_mtime.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": source,
                        "execution_result": None,
                        "execution_timestamp": None,
                        "error_log": None,
                        "file_path": str(version_file),
                        "user_prompt": None  # Can't recover prompt from file, set to None
                    }
                    
                    versions.append(version_entry)
                    existing_version_numbers.add(version_num)
                    missing_versions.append(version_num)
                    print(f"DEBUG: Found missing version {version_num} in file {filename}, adding to metadata")
            except Exception as e:
                print(f"Warning: Error processing {version_file} in _ensure_all_file_versions_in_metadata: {e}")
        
        if missing_versions:
            # Sort versions by version number
            versions.sort(key=lambda x: x["version"])
            # Save immediately to ensure disk metadata is up-to-date
            self._save_metadata()
            print(f"DEBUG: Added {len(missing_versions)} missing version(s) {missing_versions} to metadata before merge")
    
    def recover_missing_versions(self, script_name: str) -> int:
        """
        Recover missing versions by scanning version files in the directory
        and adding them to metadata if they're missing.
        
        CRITICAL: This function preserves existing user_prompt values and only
        adds missing versions. It does NOT overwrite existing prompts.
        
        Args:
            script_name: Name of the script to recover versions for
            
        Returns:
            Number of versions recovered
        """
        recovered_count = 0
        
        # Load current metadata from disk to preserve any existing prompts
        disk_metadata = self._load_metadata()
        if script_name in disk_metadata:
            # Merge disk metadata with in-memory to preserve prompts
            disk_versions = {v["version"]: v for v in disk_metadata[script_name].get("versions", [])}
            for disk_v in disk_versions.values():
                # Only add if not already in memory, preserving prompts
                if script_name not in self.metadata:
                    self.metadata[script_name] = {"versions": []}
                existing = next((v for v in self.metadata[script_name]["versions"] if v["version"] == disk_v["version"]), None)
                if not existing:
                    self.metadata[script_name]["versions"].append(disk_v.copy())
        
        if script_name not in self.metadata:
            self.metadata[script_name] = {"versions": []}
        
        versions = self.metadata[script_name]["versions"]
        existing_version_numbers = {v["version"] for v in versions}
        
        # Scan for version files matching the script name pattern
        if script_name.endswith('.py'):
            pattern = f"{script_name}_v*.py"
        else:
            pattern = f"{script_name}_v*.m"
        
        for version_file in self.versions_dir.glob(pattern):
            try:
                # Extract version number from filename
                filename = version_file.name
                if script_name.endswith('.py'):
                    version_str = filename.replace(f"{script_name}_v", "").replace(".py", "")
                else:
                    version_str = filename.replace(f"{script_name}_v", "").replace(".m", "")
                
                try:
                    version_num = int(version_str)
                except ValueError:
                    print(f"Warning: Could not parse version number from {filename}")
                    continue
                
                # Check if this version is already in metadata
                if version_num not in existing_version_numbers:
                    # Read file modification time as timestamp
                    file_stat = version_file.stat()
                    file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    # Determine source
                    source = "llm_modification"
                    if version_num == 1:
                        source = "initial_copy"
                    
                    # Create version entry - user_prompt will be None initially
                    # but can be updated later via update_user_prompt method
                    version_entry = {
                        "version": version_num,
                        "timestamp": file_mtime.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": source,
                        "execution_result": None,
                        "execution_timestamp": None,
                        "error_log": None,
                        "file_path": str(version_file),
                        "user_prompt": None  # Will be set via dedicated method if available
                    }
                    
                    versions.append(version_entry)
                    existing_version_numbers.add(version_num)
                    recovered_count += 1
                    print(f"DEBUG: Recovered version {version_num} for {script_name} from file {filename}")
            except Exception as e:
                print(f"Warning: Error processing {version_file}: {e}")
        
        if recovered_count > 0:
            # Sort versions by version number
            versions.sort(key=lambda x: x["version"])
            # CRITICAL: Only save if we added new versions, and preserve existing prompts
            self._save_metadata()
            print(f"Recovered {recovered_count} missing version(s) for {script_name}")
        
        return recovered_count
    
    def clear_all_history(self):
        """Clear all version history (delete all version files and reset metadata)"""
        try:
            # Delete all version files in the versions directory
            if self.versions_dir.exists():
                for file_path in self.versions_dir.glob("*_v*.py"):
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"Warning: Could not delete {file_path}: {e}")
                for file_path in self.versions_dir.glob("*_v*.m"):
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"Warning: Could not delete {file_path}: {e}")
            
            # Reset metadata
            self.metadata = {}
            self._save_metadata()
            print("Version history cleared successfully.")
        except Exception as e:
            print(f"Error clearing version history: {e}")

