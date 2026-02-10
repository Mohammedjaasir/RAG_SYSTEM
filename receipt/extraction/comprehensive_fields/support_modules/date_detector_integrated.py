"""
Integrated Date Detector - Comprehensive date detection with multiple patterns
"""
import re
from datetime import datetime
from typing import Optional, List, Tuple


class IntegratedReceiptDateDetector:
    """Integrated receipt date detector with comprehensive patterns and formats."""
    
    def __init__(self):
        self.date_patterns = self._build_enhanced_patterns()
        self.date_formats = self._build_comprehensive_formats()

    def _build_enhanced_patterns(self) -> List[Tuple[re.Pattern, str, str]]:
        """Build comprehensive date patterns with priority ordering."""
        # HIGH PRIORITY: Explicit and unambiguous patterns
        high_priority = [
            (r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)'?\s*\d{2,4}\b", "DD Mon'YY"),
            (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{2,4})?\b', 'Month DD, YYYY?'),
            (r'\b(?:date|invoice date|bill date|transaction date|receipt date|purchase date|sale date|dt|dated)[:\s]*\d{1,2}[-\/\.]\d{1,2}[-\/\.]\d{2,4}\b', 'Labelled D/M/Y'),
            (r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b', 'ISO timestamp'),
            (r'\b\d{4}-\d{2}-\d{2}\b', 'ISO date'),
            (r'\b(?:19|20)\d{6}\b', 'YYYYMMDD'),
            (r'\b\d{1,2}[-\/\.]\d{1,2}[-\/\.]\d{4}\b', 'D/M/YYYY_4digit'),
        ]
        
        # MEDIUM PRIORITY: Less explicit but still reliable
        medium_priority = [
            (r'\b\d{1,2}[-\/\.]\d{1,2}[-\/\.]\d{2}\b', 'D/M/Y_2digit'),
            (r'\b\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\d{2,4}\b', 'DDMonYYYY_no_sep'),
            (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\d{1,2}\d{2,4}\b', 'MonDDYYYY_no_sep'),
        ]
        
        # LOW PRIORITY: Ambiguous patterns
        low_priority = [
            (r'\b\d{6}\b', '6-digit'),
        ]
        
        # Compile patterns with priority metadata
        patterns = []
        for priority, pattern_list in [('high', high_priority), ('medium', medium_priority), ('low', low_priority)]:
            for pattern, desc in pattern_list:
                patterns.append((re.compile(pattern, re.IGNORECASE), desc, priority))
        
        return patterns

    def _build_comprehensive_formats(self) -> List[str]:
        """Build comprehensive date formats."""
        return [
            '%d-%m-%Y', '%d-%m-%y',
            '%d/%m/%Y', '%d.%m.%Y', '%d/%m/%y', '%d.%m.%y',
            '%Y-%m-%d', '%Y%m%d', '%y%m%d',
            '%d %b %Y', '%d %b %y', '%d %B %Y', '%d %B %y',
            "%d %b'%y", "%d %b'%Y", "%d %b '%y", "%d %b '%Y",
            '%b %d %Y', '%B %d %Y', '%B %d, %Y',
            '%d%m%Y', '%d%m%y',
            '%d-%b-%y', '%d-%b-%Y',
            '%d/%b/%Y', '%d.%b.%Y', '%d %b, %Y',
            '%Y-%m-%dT%H:%M:%S'
        ]

    def _clean(self, s: str) -> str:
        """Enhanced text cleaning."""
        s = s.strip()
        s = s.replace("'", "'").replace("`", "'")
        s = re.sub(r'^(date|dt|dated|invoice date|bill date|transaction date|receipt date|purchase date|sale date)[:\s\-]*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'[\.,;:\)\]]+$', '', s)
        s = re.sub(r'^[\(\[]+', '', s)
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r"([A-Za-z]+)'(\d{2,4})", r"\1 '\2", s)
        s = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', s, flags=re.IGNORECASE)
        s = re.sub(r'([A-Za-z]{3,})(?=\d)', r'\1 ', s)
        s = re.sub(r'(?<=\d)(?=[A-Za-z]{3,})', ' ', s)
        return s.strip()

    def _try_parse(self, s: str) -> Optional[datetime]:
        """Advanced date parsing with comprehensive format support."""
        s = self._clean(s)
        s = s.replace(',', '')
        for fmt in self.date_formats:
            try:
                dt = datetime.strptime(s, fmt)
                if 1900 <= dt.year <= 2100:
                    return dt
            except Exception:
                continue
        
        # Fallback: digit extraction for 6/8-digit numbers
        digits = re.sub(r'\D', '', s)
        if len(digits) == 8:
            for fmt in ('%Y%m%d', '%d%m%Y'):
                try:
                    return datetime.strptime(digits, fmt)
                except Exception:
                    pass
        if len(digits) == 6:
            try:
                mm = int(digits[2:4])
                if 1 <= mm <= 12:
                    return datetime.strptime(digits, '%y%m%d')
            except Exception:
                pass
            try:
                return datetime.strptime(digits, '%d%m%y')
            except Exception:
                pass
        return None

    def detect_date(self, text: str, debug: bool = False) -> str:
        """Comprehensive date detection with smart search strategy and priority-based matching."""
        text = text or ''
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        
        # Smart search strategy: labeled lines first, then all lines, then full text
        labelled = [ln for ln in lines if re.search(r'\b(date|invoice|bill|transaction|receipt|purchase|sale|dt|dated)\b', ln, re.IGNORECASE)]
        search_order = labelled + lines + [text]
        
        # Collect all potential matches with priority
        all_matches = []
        seen = set()

        for chunk in search_order:
            for pat, desc, priority in self.date_patterns:
                for m in pat.finditer(chunk):
                    raw = m.group(0)
                    if raw in seen:
                        continue
                    seen.add(raw)
                    dt = self._try_parse(raw)
                    if debug:
                        print(f"[MATCH] {desc} ({priority}): '{raw}' -> {dt}")
                    if dt:
                        all_matches.append({
                            'date': dt,
                            'raw': raw,
                            'desc': desc,
                            'priority': priority,
                            'source': chunk[:50] + '...' if len(chunk) > 50 else chunk
                        })
        
        if not all_matches:
            return "No date found"
        
        # Sort by priority (high > medium > low) and by year
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        all_matches.sort(key=lambda x: (
            priority_order[x['priority']],
            x['date'].year if x['date'].year >= 2020 else x['date'].year + 100,
            -len(x['raw'])
        ), reverse=True)
        
        return all_matches[0]['date'].strftime('%Y-%m-%d')