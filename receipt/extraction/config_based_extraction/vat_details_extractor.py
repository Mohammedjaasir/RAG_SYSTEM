import re

def extract_vat_rows(text, config):
    lines = text.split('\n')
    extracted_rows = []
    
    # Handle list vs dict wrapper
    strategies = config if isinstance(config, list) else config.get("strategies", config.get("vat_details_config", []))
    excludes = config.get("excludes", []) if isinstance(config, dict) else []
    
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line or any(ex in line.lower() for ex in excludes):
            continue

        for rule in strategies:
            rule_type = rule.get("type", "regex_only")
            regex = rule.get("regex", "")
            mapping = rule.get("mapping", {})
            
            if not regex: continue

            match = None
            
            # --- STRATEGY: STANDARD REGEX ---
            if rule_type == "regex_only":
                match = re.search(regex, clean_line, re.IGNORECASE)

            # --- STRATEGY: LOOK AHEAD (Trigger on current, Capture from next) ---
            # Used for: "VAT" keyword on one line, Amount on the next
            elif rule_type == "look_ahead" and i + 1 < len(lines):
                if re.search(regex, clean_line, re.IGNORECASE):
                    target_regex = rule.get("target_regex", "")
                    match = re.search(target_regex, lines[i+1].strip(), re.IGNORECASE)

            # --- STRATEGY: LOOK BEHIND (Capture on current, Validate previous) ---
            # Used for: Tables where we need to check the header (Net Vat Total)
            elif rule_type == "look_behind" and i > 0:
                match = re.search(regex, clean_line, re.IGNORECASE)
                if match:
                    context_regex = rule.get("context_regex", "")
                    if not re.search(context_regex, lines[i-1].strip(), re.IGNORECASE):
                        match = None

            if match:
                row_data = {}
                is_valid = True
                
                for field_name, map_val in mapping.items():
                    try:
                        # FEATURE: CONSTANT VALUE (e.g. "amount": 0.0)
                        if isinstance(map_val, (int, float)) and not isinstance(map_val, bool) and match.lastindex is None:
                             # Should rarely happen with re.search unless strictly 0 groups, 
                             # but primarily for manual constants if we want to allow skipping groups.
                             # Actually, easier logic: checks if map_val is a float/int 
                             # AND not a group index (which are ints). 
                             # Distinguishing 0.0 (float) from 0 (group) is key.
                             if isinstance(map_val, float): 
                                 row_data[field_name] = map_val
                                 continue
                        
                        # FEATURE: CONSTANT VIA TYPE CHECK
                        # If the json has 0.0, python reads it as float. Group indices are ints.
                        if isinstance(map_val, float):
                            row_data[field_name] = map_val
                            continue

                        # STANDARD GROUP EXTRACTION
                        raw_value = match.group(map_val)
                        clean_val = raw_value.replace('£', '').replace('$', '').strip()
                        
                        if field_name in ["rate", "amount", "net_amount", "gross_amount"]:
                            # Remove internal spaces (e.g. "0. 00")
                            clean_val = clean_val.replace(' ', '').replace(',', '.')
                            row_data[field_name] = float(clean_val)
                        else:
                            row_data[field_name] = clean_val
                            
                    except (ValueError, IndexError, AttributeError):
                        is_valid = False
                        break
                
                if is_valid:
                    extracted_rows.append(row_data)
                    break 

    return extracted_rows