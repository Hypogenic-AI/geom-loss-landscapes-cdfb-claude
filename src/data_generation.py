"""
Data generation for propositional logic theorem proving.
Generates (premises, conclusion) pairs with optional chain-of-thought derivations.
"""

import random
import itertools
from typing import List, Tuple, Dict

# Propositional logic operations
VARS = ['p', 'q', 'r', 's']


def eval_expr(expr: str, assignment: Dict[str, bool]) -> bool:
    """Evaluate a propositional logic expression given variable assignments."""
    # Replace variables with truth values
    s = expr
    for var, val in assignment.items():
        s = s.replace(var, str(val))
    s = s.replace('NOT True', 'False').replace('NOT False', 'True')
    # Iteratively simplify
    for _ in range(10):
        s = s.replace('NOT True', 'False').replace('NOT False', 'True')
        s = s.replace('True AND True', 'True')
        s = s.replace('True AND False', 'False')
        s = s.replace('False AND True', 'False')
        s = s.replace('False AND False', 'False')
        s = s.replace('True OR True', 'True')
        s = s.replace('True OR False', 'True')
        s = s.replace('False OR True', 'True')
        s = s.replace('False OR False', 'False')
        s = s.replace('True IMPLIES True', 'True')
        s = s.replace('True IMPLIES False', 'False')
        s = s.replace('False IMPLIES True', 'True')
        s = s.replace('False IMPLIES False', 'True')
    return s.strip() == 'True'


def is_tautological_consequence(premises: List[str], conclusion: str) -> bool:
    """Check if conclusion follows from premises via truth table."""
    for vals in itertools.product([True, False], repeat=len(VARS)):
        assignment = dict(zip(VARS, vals))
        if all(eval_expr(p, assignment) for p in premises):
            if not eval_expr(conclusion, assignment):
                return False
    return True


def generate_simple_expr(depth: int = 0, max_depth: int = 1) -> str:
    """Generate a random propositional expression."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.5):
        return random.choice(VARS[:3])  # Use p, q, r for simpler expressions

    op = random.choice(['AND', 'OR', 'IMPLIES', 'NOT'])
    if op == 'NOT':
        sub = generate_simple_expr(depth + 1, max_depth)
        return f'NOT {sub}'
    else:
        left = random.choice(VARS[:3])
        right = random.choice(VARS[:3])
        return f'{left} {op} {right}'


def generate_derivation_steps(premises: List[str], conclusion: str) -> List[str]:
    """Generate chain-of-thought derivation steps."""
    steps = []

    # Step 1: Restate premises
    steps.append(f"Given: {'; '.join(premises)}")

    # Step 2: Apply inference rules
    if len(premises) >= 2:
        # Try modus ponens pattern
        for i, p in enumerate(premises):
            if 'IMPLIES' in p:
                parts = p.split(' IMPLIES ')
                if len(parts) == 2:
                    antecedent, consequent = parts
                    for j, p2 in enumerate(premises):
                        if i != j and p2.strip() == antecedent.strip():
                            steps.append(f"By modus ponens on premise {j+1} and {i+1}: {consequent.strip()}")
                            break

    # Step 3: Simplification steps
    for p in premises:
        if 'AND' in p:
            parts = p.split(' AND ')
            if len(parts) == 2:
                steps.append(f"By conjunction elimination: {parts[0].strip()}")
                steps.append(f"By conjunction elimination: {parts[1].strip()}")
                break

    # Step 4: Derive conclusion
    steps.append(f"Therefore: {conclusion}")

    return steps


def generate_dataset(n_samples: int = 1000, seed: int = 42) -> Tuple[List, List]:
    """
    Generate theorem proving dataset.
    Returns (direct_data, cot_data) where each is a list of (input_str, output_str).
    """
    random.seed(seed)

    direct_data = []
    cot_data = []

    templates = [
        # Modus ponens: p, p -> q |- q
        lambda: (['p', 'p IMPLIES q'], 'q'),
        # Modus tollens: NOT q, p -> q |- NOT p
        lambda: (['NOT q', 'p IMPLIES q'], 'NOT p'),
        # Hypothetical syllogism: p -> q, q -> r |- p -> r
        lambda: (['p IMPLIES q', 'q IMPLIES r'], 'p IMPLIES r'),
        # Conjunction intro: p, q |- p AND q
        lambda: (['p', 'q'], 'p AND q'),
        # Conjunction elim: p AND q |- p
        lambda: (['p AND q'], 'p'),
        # Disjunction intro: p |- p OR q
        lambda: (['p'], 'p OR q'),
        # Disjunctive syllogism: p OR q, NOT p |- q
        lambda: (['p OR q', 'NOT p'], 'q'),
        # Double negation: NOT NOT p |- p
        lambda: (['NOT NOT p'], 'p'),
        # Chain: p, p -> q, q -> r |- r
        lambda: (['p', 'p IMPLIES q', 'q IMPLIES r'], 'r'),
        # Complex: p AND q, p IMPLIES r |- r
        lambda: (['p AND q', 'p IMPLIES r'], 'r'),
    ]

    # Variable substitution patterns
    var_subs = [
        {'p': 'p', 'q': 'q', 'r': 'r'},
        {'p': 'q', 'q': 'r', 'r': 's'},
        {'p': 'r', 'q': 'p', 'r': 'q'},
        {'p': 'p', 'q': 's', 'r': 'q'},
        {'p': 'q', 'q': 'p', 'r': 's'},
        {'p': 's', 'q': 'q', 'r': 'p'},
    ]

    for _ in range(n_samples):
        template = random.choice(templates)
        premises, conclusion = template()

        # Apply variable substitution
        sub = random.choice(var_subs)
        premises = [apply_sub(p, sub) for p in premises]
        conclusion = apply_sub(conclusion, sub)

        # Create input string
        input_str = ' ; '.join(premises) + ' |- '

        # Direct output: just the conclusion
        direct_output = conclusion

        # CoT output: step-by-step derivation
        steps = generate_derivation_steps(premises, conclusion)
        cot_output = ' | '.join(steps)

        direct_data.append((input_str, direct_output))
        cot_data.append((input_str, cot_output))

    return direct_data, cot_data


def apply_sub(expr: str, sub: Dict[str, str]) -> str:
    """Apply variable substitution to expression."""
    # Use placeholder to avoid double-substitution
    result = expr
    placeholders = {}
    for old, new in sub.items():
        placeholder = f'__PLACEHOLDER_{old}__'
        result = result.replace(old, placeholder)
        placeholders[placeholder] = new
    for placeholder, new in placeholders.items():
        result = result.replace(placeholder, new)
    return result


# Tokenizer for our simple language
class SimpleTokenizer:
    """Character-level tokenizer for propositional logic expressions."""

    def __init__(self):
        # Build vocabulary from all possible tokens
        self.tokens = ['<pad>', '<bos>', '<eos>', '<sep>']
        # Add single characters and keywords
        keywords = ['AND', 'OR', 'NOT', 'IMPLIES', 'Given:', 'By', 'modus',
                     'ponens', 'on', 'premise', 'conjunction', 'elimination:',
                     'Therefore:', 'disjunction', 'syllogism', 'tollens',
                     'hypothetical', 'negation', 'double']
        self.tokens.extend(keywords)
        # Add variables and punctuation
        for c in 'pqrs|-;,. 0123456789':
            if c not in self.tokens:
                self.tokens.append(c)

        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.vocab_size = len(self.tokens)
        self.pad_id = self.token2id['<pad>']
        self.bos_id = self.token2id['<bos>']
        self.eos_id = self.token2id['<eos>']
        self.sep_id = self.token2id['<sep>']

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids using greedy longest-match."""
        ids = [self.bos_id]
        i = 0
        while i < len(text):
            # Try longest match first
            matched = False
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.token2id:
                    ids.append(self.token2id[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                # Skip unknown character
                i += 1
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text."""
        tokens = []
        for id in ids:
            if id in (self.pad_id, self.bos_id, self.eos_id):
                continue
            tokens.append(self.id2token.get(id, '?'))
        return ''.join(tokens)

    def pad_sequence(self, ids: List[int], max_len: int) -> List[int]:
        """Pad or truncate sequence to max_len."""
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [self.pad_id] * (max_len - len(ids))


if __name__ == '__main__':
    direct_data, cot_data = generate_dataset(n_samples=10, seed=42)
    tok = SimpleTokenizer()
    print(f"Vocab size: {tok.vocab_size}")
    print(f"\nExample direct data:")
    for inp, out in direct_data[:3]:
        print(f"  Input:  {inp}")
        print(f"  Output: {out}")
        print(f"  Encoded input len: {len(tok.encode(inp))}")
        print(f"  Encoded output len: {len(tok.encode(out))}")
        print()
    print(f"Example CoT data:")
    for inp, out in cot_data[:3]:
        print(f"  Input:  {inp}")
        print(f"  Output: {out}")
        print()
