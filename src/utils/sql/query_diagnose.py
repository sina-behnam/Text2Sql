"""
SQL Query Diagnostic Tool
Identifies queries that might hang or cause timeouts
"""

import json
import re
from typing import List, Tuple, Dict


def analyze_query_complexity(sql: str) -> Dict[str, any]:
    """
    Analyze SQL query for potential issues that could cause hanging.
    
    Returns dictionary with:
    - risk_level: 'low', 'medium', 'high', 'critical'
    - issues: list of potential problems
    - metrics: various query metrics
    """
    if not sql:
        return {'risk_level': 'unknown', 'issues': ['Empty query'], 'metrics': {}}
    
    sql_upper = sql.upper()
    issues = []
    risk_score = 0
    
    metrics = {
        'length': len(sql),
        'joins': sql_upper.count('JOIN'),
        'subqueries': sql_upper.count('SELECT') - 1,  # -1 for main SELECT
        'unions': sql_upper.count('UNION'),
        'group_by': 1 if 'GROUP BY' in sql_upper else 0,
        'order_by': 1 if 'ORDER BY' in sql_upper else 0,
        'distinct': 1 if 'DISTINCT' in sql_upper else 0,
        'aggregates': (sql_upper.count('COUNT') + sql_upper.count('SUM') + 
                      sql_upper.count('AVG') + sql_upper.count('MAX') + 
                      sql_upper.count('MIN')),
    }
    
    # Check for cartesian products (JOIN without ON)
    # Pattern: table1, table2, table3 or JOIN without ON
    if ',' in sql and 'FROM' in sql_upper:
        # Count comma-separated tables in FROM clause
        from_match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP|ORDER|LIMIT|$)', sql_upper, re.DOTALL)
        if from_match:
            from_clause = from_match.group(1)
            table_count = from_clause.count(',') + 1
            if table_count > 1:
                issues.append(f'Potential cartesian product: {table_count} tables with comma syntax')
                risk_score += 40
    
    # Check for JOIN without ON
    join_pattern = r'JOIN\s+\w+\s+(?!ON)'
    if re.search(join_pattern, sql_upper):
        issues.append('JOIN without ON clause detected')
        risk_score += 50
    
    # Multiple joins
    if metrics['joins'] >= 5:
        issues.append(f"Many joins ({metrics['joins']})")
        risk_score += 15
    elif metrics['joins'] >= 3:
        risk_score += 5
    
    # Deep nesting
    if metrics['subqueries'] >= 4:
        issues.append(f"Deep nesting ({metrics['subqueries']} subqueries)")
        risk_score += 20
    elif metrics['subqueries'] >= 2:
        risk_score += 5
    
    # Missing LIMIT on large operations
    if 'LIMIT' not in sql_upper:
        if metrics['joins'] >= 2 or metrics['subqueries'] >= 2:
            issues.append('No LIMIT clause with complex query')
            risk_score += 10
    
    # Complex aggregations without proper indexing hint
    if metrics['group_by'] and metrics['joins'] >= 2:
        risk_score += 10
    
    # DISTINCT on multiple joins
    if metrics['distinct'] and metrics['joins'] >= 3:
        issues.append('DISTINCT with multiple joins (can be slow)')
        risk_score += 15
    
    # Very long queries
    if metrics['length'] > 1000:
        issues.append(f'Very long query ({metrics['length']} chars)')
        risk_score += 10
    
    # Self-joins (same table multiple times)
    tables = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql_upper)
    table_list = [t[0] or t[1] for t in tables]
    unique_tables = set(table_list)
    if len(table_list) > len(unique_tables):
        duplicates = len(table_list) - len(unique_tables)
        issues.append(f'Self-join detected ({duplicates} duplicate tables)')
        risk_score += 5 * duplicates
    
    # Determine risk level
    if risk_score >= 60:
        risk_level = 'critical'
    elif risk_score >= 35:
        risk_level = 'high'
    elif risk_score >= 15:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'issues': issues if issues else ['None detected'],
        'metrics': metrics
    }


def diagnose_dataset(json_file: str, output_file: str = None):
    """
    Diagnose all SQL queries in the dataset.
    
    Args:
        json_file: Path to JSON file with extracted SQL queries
        output_file: Optional path to save detailed report
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"SQL QUERY DIAGNOSTICS")
    print(f"{'='*80}\n")
    
    risk_counts = {'critical': [], 'high': [], 'medium': [], 'low': [], 'unknown': []}
    all_diagnostics = {}
    
    for entry_id, entry in data.items():
        extracted_sql = entry.get('extracted_sql', '')
        
        if not extracted_sql:
            risk_counts['unknown'].append(entry_id)
            continue
        
        diagnosis = analyze_query_complexity(extracted_sql)
        risk_level = diagnosis['risk_level']
        risk_counts[risk_level].append(entry_id)
        
        all_diagnostics[entry_id] = {
            'question': entry.get('question', ''),
            'database': entry.get('database', ''),
            'extracted_sql': extracted_sql,
            'diagnosis': diagnosis
        }
    
    # Print summary
    total = len(data)
    print(f"Total queries analyzed: {total}\n")
    print(f"{'Risk Level':<15} {'Count':<10} {'Percentage'}")
    print(f"{'-'*80}")
    
    for level in ['critical', 'high', 'medium', 'low', 'unknown']:
        count = len(risk_counts[level])
        pct = (count / total * 100) if total > 0 else 0
        symbol = '🔴' if level == 'critical' else '🟠' if level == 'high' else '🟡' if level == 'medium' else '🟢'
        print(f"{symbol} {level.capitalize():<13} {count:<10} {pct:>6.1f}%")
    
    # Show examples of problematic queries
    if risk_counts['critical']:
        print(f"\n{'='*80}")
        print(f"🔴 CRITICAL RISK QUERIES (likely to hang)")
        print(f"{'='*80}\n")
        
        for i, entry_id in enumerate(risk_counts['critical'][:5], 1):
            diag = all_diagnostics[entry_id]
            print(f"{i}. Entry {entry_id}")
            print(f"   Database: {diag['database']}")
            print(f"   Question: {diag['question'][:70]}...")
            print(f"   Risk Score: {diag['diagnosis']['risk_score']}")
            print(f"   Issues: {', '.join(diag['diagnosis']['issues'])}")
            print(f"   SQL: {diag['extracted_sql'][:100]}...")
            print()
    
    if risk_counts['high']:
        print(f"\n{'='*80}")
        print(f"🟠 HIGH RISK QUERIES (may timeout)")
        print(f"{'='*80}\n")
        
        for i, entry_id in enumerate(risk_counts['high'][:3], 1):
            diag = all_diagnostics[entry_id]
            print(f"{i}. Entry {entry_id}")
            print(f"   Database: {diag['database']}")
            print(f"   Risk Score: {diag['diagnosis']['risk_score']}")
            print(f"   Issues: {', '.join(diag['diagnosis']['issues'])}")
            print(f"   SQL: {diag['extracted_sql'][:100]}...")
            print()
    
    # Save detailed report
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_diagnostics, f, indent=2)
        print(f"\n✓ Detailed diagnostics saved to: {output_file}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if risk_counts['critical']:
        print(f"⚠️  {len(risk_counts['critical'])} critical queries detected!")
        print(f"   - These queries likely have cartesian products or missing JOIN conditions")
        print(f"   - Use shorter timeout (5-10 seconds) for these")
        print(f"   - Consider skipping or pre-validating these queries\n")
    
    if risk_counts['high']:
        print(f"⚠️  {len(risk_counts['high'])} high-risk queries detected")
        print(f"   - Use 30-60 second timeout for these")
        print(f"   - Monitor execution time closely\n")
    
    if risk_counts['medium']:
        print(f"ℹ️  {len(risk_counts['medium'])} medium-risk queries")
        print(f"   - Should execute fine with 30 second timeout\n")
    
    print(f"{'='*80}\n")
    
    return all_diagnostics, risk_counts


if __name__ == "__main__":
    # Run diagnostic on extracted queries
    json_file = '/mnt/user-data/outputs/extracted_sql_queries.json'
    output_file = '/mnt/user-data/outputs/sql_diagnostics_report.json'
    
    diagnostics, risk_counts = diagnose_dataset(json_file, output_file)
    
    # Export list of problematic query IDs
    problematic_ids = risk_counts['critical'] + risk_counts['high']
    
    if problematic_ids:
        with open('/mnt/user-data/outputs/problematic_query_ids.txt', 'w') as f:
            f.write("# Query IDs likely to cause timeouts\n")
            f.write("# Format: entry_id\n\n")
            for qid in problematic_ids:
                f.write(f"{qid}\n")
        
        print(f"✓ List of problematic IDs saved to: problematic_query_ids.txt")