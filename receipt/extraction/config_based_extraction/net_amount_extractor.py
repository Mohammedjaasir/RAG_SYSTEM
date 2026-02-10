import re

def extract_net(lines, strategies):
    """
    Scans lines bottom-to-top.
    Collects candidates based on regex strategies (with optional context keywords).
    Returns the maximum valid float found.
    """
    candidates = []
    # Pre-calculate reversed lines for performance
    reversed_lines = list(reversed(lines))
    
    for rule in strategies:
        rule_type = rule.get("type", "regex_only")
        regex = rule.get("regex", "")
        keywords = rule.get("keywords", [])
        
        # Compile regex once for speed
        try:
            pattern = re.compile(regex, re.IGNORECASE)
        except re.error:
            continue

        for line in reversed_lines:
            # 1. Context Check (Skip line if keywords missing)
            if rule_type == "context_line":
                if not any(k in line.lower() for k in keywords):
                    continue

            # 2. Find Matches
            matches = pattern.finditer(line)
            
            for m in matches:
                try:
                    clean_str = ""
                    
                    # CASE A: Regex has Capture Groups (e.g. "16" ... "00")
                    # Join valid groups with a dot. 
                    # This covers the old "lastindex == 2" logic AND "lastindex == 1" logic automatically.
                    if m.groups():
                        valid_groups = [g for g in m.groups() if g]
                        clean_str = ".".join(valid_groups)
                        
                    # CASE B: Regex matches whole string
                    else:
                        clean_str = m.group(0)

                    # 3. GENERIC CLEANING
                    # a. Standardize dots/commas
                    clean_str = clean_str.replace(',', '.')
                    
                    # b. Remove anything that isn't a digit or dot (Fixes spaces like "20 . 84")
                    clean_str = re.sub(r"[^\d.]", "", clean_str)
                    
                    # c. Fix Double Dots (Common OCR error "20..84" -> "20.84")
                    if clean_str.count('.') > 1:
                        # Split by dot and keep the last part as decimal
                        parts = clean_str.split('.')
                        clean_str = "".join(parts[:-1]) + '.' + parts[-1]
                    
                    val = float(clean_str)
                    candidates.append(val)

                except ValueError:
                    continue

    # Final Selection: Return the Maximum value found
    if candidates:
        return f"{max(candidates):.2f}"
            
    return None