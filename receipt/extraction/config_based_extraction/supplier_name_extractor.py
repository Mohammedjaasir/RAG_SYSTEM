import re

def extract_supplier(lines, config):
    """
    Extracts supplier name by iterating through a prioritized list of strategies.
    strategies: List of dicts defining 'type' and 'regex'/'list'.
    """
    # 1. Setup & Pre-processing
    max_lines = config.get("max_lines", 8) 
    excludes = [ex.lower() for ex in config.get("excludes", [])]
    
    # Flatten known suppliers list if provided
    # We use a set for faster lookups, but keep a mapping to return original casing
    known_suppliers_map = {}
    for name in config.get("all_suppliers", []):
        known_suppliers_map[name.lower()] = name

    # Filter lines (Remove header noise, empty lines)
    candidate_lines = []
    for line in lines[:max_lines]:
        clean_line = line.strip()
        if not clean_line: continue
        
        # Skip excluded lines
        if any(ex in clean_line.lower() for ex in excludes): continue
        
        # Skip lines that are purely numeric/dates (e.g. "12/05/2023")
        if re.match(r'^[\d/.:\-\s]+$', clean_line): continue
        
        candidate_lines.append(clean_line)

    # 2. Iterate Strategies (Priority Order)
    strategies = config.get("strategies", [])
    
    for strategy in strategies:
        strat_type = strategy.get("type", "")
        
        # --- STRATEGY: EXACT MATCH (High Priority) ---
        # Checks if the line contains a known brand (e.g. "Tesco")
        if strat_type == "known_match":
            for line in candidate_lines:
                line_lower = line.lower()
                # Check for known suppliers
                # We sort keys by length desc to match "Tesco Express" before "Tesco"
                for key in sorted(known_suppliers_map.keys(), key=len, reverse=True):
                    # Word boundary check is safer: matches "Tesco" but not "Tescooooo"
                    if re.search(r'\b' + re.escape(key) + r'\b', line_lower):
                        return known_suppliers_map[key]

        # --- STRATEGY: REGEX EXTRACTION (Contextual) ---
        # Extracts a group: "Welcome to (Group 1)"
        elif strat_type == "regex_extract":
            regex = strategy.get("regex", "")
            group = strategy.get("group", 1)
            for line in candidate_lines:
                match = re.search(regex, line, re.IGNORECASE)
                if match:
                    # Return the captured group (e.g. the name after "Welcome to")
                    return match.group(group).strip().title()

        # --- STRATEGY: REGEX MATCH LINE (Pattern Recognition) ---
        # If line matches pattern (e.g. ends in "Ltd"), return the WHOLE line.
        elif strat_type == "regex_match_line":
            regex = strategy.get("regex", "")
            for line in candidate_lines:
                if re.search(regex, line, re.IGNORECASE):
                    # We found a line matching a business pattern. Return it.
                    return line.strip()

        # --- STRATEGY: FALLBACK ---
        # Just return the first valid text line we found.
        elif strat_type == "fallback_first_line":
            if candidate_lines:
                return candidate_lines[0]

    return None