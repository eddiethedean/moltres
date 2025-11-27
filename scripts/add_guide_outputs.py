#!/usr/bin/env python3
"""Run all guide code blocks and add their outputs as comments.

This script:
1. Extracts all Python code blocks from guide markdown files
2. Executes each block sequentially (maintaining context)
3. Captures stdout/stderr from each execution
4. Adds output as comments after print statements or at the end of blocks
"""

import re
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def extract_code_blocks(content: str) -> List[Tuple[int, str, str]]:
    """Extract Python code blocks from markdown content.
    
    Returns:
        List of (line_number, full_match_text, code) tuples
    """
    pattern = r"(```python\n(.*?)```)"
    blocks = []
    for match in re.finditer(pattern, content, re.DOTALL):
        lines_before = content[:match.start()].count('\n')
        full_match = match.group(1)  # The entire ```python...``` block
        code = match.group(2).strip()
        if code and not code.startswith('#'):  # Skip empty blocks and comment-only blocks
            blocks.append((lines_before + 1, full_match, code))
    return blocks


def execute_code_block(code: str, temp_dir: Path, context_vars: dict) -> Tuple[bool, str, str, dict]:
    """Execute a code block and capture output.
    
    Args:
        code: Python code to execute
        temp_dir: Temporary directory for execution
        context_vars: Dictionary of variables to inject before execution
        
    Returns:
        Tuple of (success, stdout, stderr, updated_context_vars)
    """
    # Build full code with context
    context_code = "\n".join([f"{k} = {repr(v)}" if not isinstance(v, str) or "db" not in k else f"{k} = {v}" 
                               for k, v in context_vars.items() if k != '__builtins__'])
    full_code = f"{context_code}\n{code}"
    
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=temp_dir, delete=False) as f:
        f.write(full_code)
        temp_file = Path(f.name)
    
    try:
        # Execute the code
        result = subprocess.run(
            [sys.executable, str(temp_file)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=temp_dir,
            env={**{k: str(v) for k, v in context_vars.items() if isinstance(v, (str, int, float))}, 
                 **{"PYTHONPATH": str(project_root / "src") + ":" + str(project_root)}}
        )
        
        # Try to capture variables from execution context
        # This is tricky - we'll use a simpler approach: just capture output
        success = result.returncode == 0
        return success, result.stdout, result.stderr, context_vars
        
    except subprocess.TimeoutExpired:
        return False, "", "Timeout after 30 seconds", context_vars
    except Exception as e:
        return False, "", str(e), context_vars
    finally:
        try:
            temp_file.unlink()
        except Exception:
            pass


def add_output_to_code(code: str, stdout: str, stderr: str, success: bool) -> str:
    """Add output as comments to code block.
    
    Adds output after print statements or at the end of the block.
    """
    # Check if output already exists
    if "# Output:" in code or "# Outputs:" in code:
        return code
    
    lines = code.split('\n')
    output_lines = []
    
    if stdout:
        stdout_clean = stdout.rstrip()
        if stdout_clean:
            for line in stdout_clean.split('\n'):
                line_clean = line.rstrip()
                if line_clean and not line_clean.startswith('#'):
                    # Format as comment
                    if line_clean.startswith('[{') or line_clean.startswith('{'):
                        # It's a data structure - format nicely
                        output_lines.append(f"# Output: {line_clean}")
                    else:
                        output_lines.append(f"# Output: {line_clean}")
    
    if stderr and not success:
        stderr_clean = stderr.rstrip()
        if stderr_clean and "Traceback" not in stderr_clean:  # Skip full tracebacks
            output_lines.append(f"# Error: {stderr_clean.split(chr(10))[0]}")
    
    if not output_lines:
        return code
    
    # Find insertion point - after last print statement or at end
    insert_pos = len(lines)
    
    # Look for print statements
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if 'print(' in stripped or 'print ' in stripped:
            insert_pos = i + 1
            break
    
    # Build result
    result_lines = lines[:insert_pos]
    
    # Add spacing if needed
    if insert_pos > 0 and lines[insert_pos - 1].strip():
        if not lines[insert_pos - 1].strip().startswith('#'):
            result_lines.append('')
    
    result_lines.extend(output_lines)
    result_lines.extend(lines[insert_pos:])
    
    return '\n'.join(result_lines)


def process_guide_file(guide_path: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Process a guide file and add outputs to code blocks.
    
    Returns:
        Tuple of (updated_blocks, total_blocks)
    """
    print(f"\nProcessing: {guide_path.name}")
    
    content = guide_path.read_text()
    blocks = extract_code_blocks(content)
    
    if not blocks:
        print(f"  No code blocks found")
        return 0, 0
    
    print(f"  Found {len(blocks)} code blocks")
    
    updated_blocks = 0
    context_vars = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        
        # Process blocks in reverse order to preserve positions when replacing
        for line_num, full_match, code in reversed(blocks):
            # Skip if already has output
            if "# Output:" in code or "# Outputs:" in code:
                continue
            
            # Skip comment-only or very short blocks
            if len(code.split('\n')) < 3 and not any(kw in code for kw in ['print(', 'collect()', 'head(', 'tail(']):
                continue
            
            print(f"  Line {line_num:4d}: ", end='', flush=True)
            
            success, stdout, stderr, _ = execute_code_block(code, temp_dir, context_vars)
            
            if success and stdout:
                updated_code = add_output_to_code(code, stdout, stderr, success)
                
                if updated_code != code:
                    # Escape special regex characters in full_match
                    escaped_match = re.escape(full_match)
                    # Replace in content
                    content = content.replace(full_match, f"```python\n{updated_code}\n```")
                    updated_blocks += 1
                    print(f"✓ Added output")
                else:
                    print(f"- No change needed")
            elif not success and stderr:
                # Only log meaningful errors (not just missing variables)
                if "NameError" not in stderr or "db" not in stderr.lower():
                    print(f"✗ Failed")
                else:
                    print(f"- Skipped (needs context)")
            else:
                print(f"- No output")
    
    # Write updated content
    if updated_blocks > 0 and not dry_run:
        guide_path.write_text(content)
        print(f"  ✅ Updated {updated_blocks} blocks")
    elif dry_run:
        print(f"  [DRY RUN] Would update {updated_blocks} blocks")
    
    return updated_blocks, len(blocks)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add outputs to guide code blocks")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just show what would change")
    parser.add_argument("guides", nargs="*", help="Specific guide files to process (default: all)")
    
    args = parser.parse_args()
    
    guides_dir = project_root / "guides"
    
    if args.guides:
        guide_files = [guides_dir / g if (guides_dir / g).exists() else Path(g) for g in args.guides]
        guide_files = [f for f in guide_files if f.exists()]
    else:
        guide_files = sorted(guides_dir.glob("*.md"))
        guide_files = [f for f in guide_files if f.name != "README.md"]
    
    total_updated = 0
    total_blocks = 0
    
    print(f"{'='*70}")
    print(f"Processing {len(guide_files)} guide files...")
    print(f"{'='*70}")
    
    for guide_file in guide_files:
        updated, total = process_guide_file(guide_file, dry_run=args.dry_run)
        total_updated += updated
        total_blocks += total
    
    print(f"\n{'='*70}")
    print(f"Summary: Updated {total_updated}/{total_blocks} code blocks across {len(guide_files)} guides")
    if args.dry_run:
        print(f"[DRY RUN] - No files were modified")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
