def build_prompt(formula, vars, table_fragment, col_names):
    display_formula = formula.replace("xor", "⊕").replace("imp", "→").replace("eq", "≡")
    header = " | ".join(col_names) + " | F"
    separator = "-" * len(header)
    rows = []

    for row, f in table_fragment:
        row_str = " | ".join(str(row[v]) for v in col_names)
        rows.append(f"{row_str} | {f}")

    table_str = "\n".join([header, separator] + rows)

    return f"""You are a logical reasoning assistant. 
Your task is to identify which variable corresponds to which column in a truth table.

Boolean function:
F = {display_formula}

Fragment of the truth table (columns are permuted):
{table_str}

Variables to assign: {", ".join(vars)}
Columns labels: {", ".join(col_names)}

Instructions:
1. Analyze the Boolean function for all possible inputs.
2. Match the columns in the fragment to the variables x, y, z...
3. Provide your final answer in the format: x=column_name, y=column_name...

Final answer MUST be wrapped in <answer> tags.
Example: <answer>x=A, y=B, z=C</answer>
"""