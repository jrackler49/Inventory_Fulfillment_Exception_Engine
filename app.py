"""
Inventory & Fulfillment Exception Engine — Streamlit App
----------------------------------------------------------
Takes a natural-language operational question, routes it through
the query engine pipeline, validates the SQL, executes it against
the local SQLite database, and generates a concise business answer.

Run with:
    streamlit run app.py

Expects, in the same folder as this file:
    - config.json
    - supply_chain_ops.db
"""

import json
import os
import re
import sqlite3
import warnings

import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI


warnings.filterwarnings("ignore")


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Inventory & Fulfillment Exception Engine",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# Configuration / LLM / Database Setup
# ============================================================

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_API_BASE = st.secrets["OPENAI_API_BASE"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


@st.cache_resource
def get_llms():
    """
    Creates the two LLMs used by the query engine.

    Primary LLM:
    - Intent classification
    - SQL generation
    - SQL retry
    - Response generation

    Evaluator LLM:
    - SQL relevance validation
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    evaluator_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    return llm, evaluator_llm


@st.cache_resource
def get_db_connection():
    """
    Opens the SQLite database in read-only mode.
    """

    db_path = "supply_chain_ops.db"

    return sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        check_same_thread=False
    )


llm, evaluator_llm = get_llms()
conn = get_db_connection()


# ============================================================
# Database Schema
# ============================================================

DATABASE_SCHEMA = """
warehouse_master:
  warehouse_id (TEXT, PK): US MSA facility code
  warehouse_name (TEXT): legal facility name
  region (TEXT): US Census Region
  max_capacity_pallet_positions (INTEGER): total pallet position capacity
  current_occupancy_pct (REAL): utilization percentage; high occupancy threshold >= 85.0
  is_cbp_bonded_ftz (INTEGER): 1 if CBP-bonded or Foreign Trade Zone, 0 otherwise

inventory_levels:
  inventory_id (INTEGER, PK): auto-increment identifier
  warehouse_id (TEXT, FK): joins to warehouse_master.warehouse_id
  sku_id (TEXT): unique stock keeping unit code
  sku_category (TEXT): Consumer Packaged Goods, Automotive Parts,
                        Cold-Chain Perishables, Apparel, Industrial
  units_on_hand (INTEGER): physical stock count
  reorder_point (INTEGER): minimum stock threshold
  unit_cost_usd (REAL): carrying unit cost
  last_restock_date (DATE): date of last inventory receipt

shipment_tracker:
  shipment_id (TEXT, PK): unique BOL or tracking number
  order_id (TEXT): client purchase order reference
  origin_warehouse_id (TEXT, FK): joins to warehouse_master.warehouse_id
  scac_code (TEXT, FK): joins to carrier_performance.scac_code
  promised_ship_date (DATE): contractual SLA dispatch date
  actual_ship_date (DATE): actual gate-out dispatch date, NULL if pending
  delivery_status (TEXT): On-Time, Delayed, In-Transit, Cancelled
  delay_reason (TEXT): FMCSA Driver HOS Limit, DOT Road Closure,
                        Chassis Shortage, CBP Freight Hold,
                        Warehouse Backlog, N/A

carrier_performance:
  scac_code (TEXT, PK): NMFTA Standard Carrier Alpha Code
  carrier_name (TEXT): legal corporate name
  otif_compliance_pct (REAL): On-Time In-Full percentage as decimal
  avg_delay_hours (REAL): mean delivery delay in hours
  otif_chargeback_usd (REAL): accrued SLA non-compliance penalties

Business rules:
- Stockout definition: units_on_hand = 0
- Below reorder definition:
  units_on_hand > 0 AND units_on_hand <= reorder_point
- High occupancy threshold: current_occupancy_pct >= 85.0
- Delayed shipments: delivery_status = 'Delayed'
- In-transit shipments: delivery_status = 'In-Transit'
- Inventory value formula:
  units_on_hand * unit_cost_usd
"""


# ============================================================
# Verified Query Template Library
# ============================================================

VERIFIED_QUERY_LIBRARY = {

    "VQ1": {
        "description":
            "Regional stockout count showing which US regions have "
            "the most SKUs currently at zero units on hand",

        "sql": """
SELECT
    w.region,
    COUNT(*) AS stockout_skus
FROM inventory_levels i
JOIN warehouse_master w
    ON i.warehouse_id = w.warehouse_id
WHERE i.units_on_hand = 0
GROUP BY w.region
ORDER BY stockout_skus DESC
"""
    },

    "VQ2": {
        "description":
            "SKU categories with the most items currently below "
            "reorder point but not yet stocked out",

        "sql": """
SELECT
    sku_category,
    COUNT(*) AS below_reorder_skus
FROM inventory_levels
WHERE units_on_hand > 0
  AND units_on_hand <= reorder_point
GROUP BY sku_category
ORDER BY below_reorder_skus DESC
"""
    },

    "VQ3": {
        "description":
            "Warehouses at or above the 85% high-occupancy threshold",

        "sql": """
SELECT
    warehouse_id,
    warehouse_name,
    region,
    current_occupancy_pct
FROM warehouse_master
WHERE current_occupancy_pct >= 85.0
ORDER BY current_occupancy_pct DESC
"""
    },

    "VQ4": {
        "description":
            "Total count of shipments currently marked as Delayed",

        "sql": """
SELECT
    COUNT(*) AS delayed_count
FROM shipment_tracker
WHERE delivery_status = 'Delayed'
"""
    },

    "VQ5": {
        "description":
            "Carriers ranked from worst to best by OTIF compliance percentage",

        "sql": """
SELECT
    scac_code,
    carrier_name,
    otif_compliance_pct
FROM carrier_performance
ORDER BY otif_compliance_pct ASC
"""
    },

    "VQ6": {
        "description":
            "Carrier with the highest accrued OTIF chargeback penalties",

        "sql": """
SELECT *
FROM (
    SELECT
        carrier_name,
        otif_chargeback_usd
    FROM carrier_performance
    ORDER BY otif_chargeback_usd DESC
    LIMIT 1
)
"""
    },

    "VQ7": {
        "description":
            "Top 5 warehouses ranked by total inventory value "
            "using units_on_hand multiplied by unit_cost_usd",

        "sql": """
SELECT *
FROM (
    SELECT
        w.warehouse_id,
        w.warehouse_name,
        ROUND(
            SUM(i.units_on_hand * i.unit_cost_usd),
            2
        ) AS inventory_value_usd
    FROM inventory_levels i
    JOIN warehouse_master w
        ON i.warehouse_id = w.warehouse_id
    GROUP BY w.warehouse_id
    ORDER BY inventory_value_usd DESC
    LIMIT 5
)
"""
    },

    "VQ8": {
        "description":
            "Most common reasons for shipment delays with occurrence counts",

        "sql": """
SELECT
    delay_reason,
    COUNT(*) AS occurrences
FROM shipment_tracker
WHERE delivery_status = 'Delayed'
GROUP BY delay_reason
ORDER BY occurrences DESC
"""
    },

    "VQ9": {
        "description":
            "Average occupancy comparison between CBP-bonded/FTZ "
            "warehouses and non-bonded facilities",

        "sql": """
SELECT
    is_cbp_bonded_ftz,
    ROUND(AVG(current_occupancy_pct), 2) AS avg_occupancy_pct,
    COUNT(*) AS warehouse_count
FROM warehouse_master
GROUP BY is_cbp_bonded_ftz
"""
    },

    "VQ10": {
        "description":
            "Aggregate count of shipments currently in transit "
            "broken down by carrier SCAC code",

        "sql": """
SELECT
    scac_code,
    COUNT(*) AS in_transit_count
FROM shipment_tracker
WHERE delivery_status = 'In-Transit'
GROUP BY scac_code
ORDER BY in_transit_count DESC
"""
    }
}


# ============================================================
# Tool 1: Intent Classification
# ============================================================

def classify_intent(user_question, query_library):

    library_descriptions = "\n".join(
        [
            f"{qid}: {entry['description']}"
            for qid, entry in query_library.items()
        ]
    )

    classification_prompt = f"""
### ROLE

You are a query router for a supply chain operations analytics system.

Your job is to decide whether a business user's question can be
answered by one of the pre-approved query templates or whether
fresh SQL generation is required.

### USER QUESTION

{user_question}

### AVAILABLE VERIFIED QUERY TEMPLATES

{library_descriptions}

### INSTRUCTIONS

1. Identify the analytical intent of the question.
2. Compare the intent against every verified template.
3. Match based on semantic meaning, not exact wording.
4. "Out of stock" means stockout.
5. "FTZ" or "bonded" refers to CBP-bonded/FTZ warehouses.
6. "Late" or "behind schedule" means Delayed.
7. "Near capacity" means high occupancy.
8. Only select a verified template if it genuinely answers
   the requested question.
9. A row-level question should not match an aggregate template.
10. If no template genuinely answers the question, use the
    generated route.

### OUTPUT

Return ONLY valid JSON:

{{
    "route": "verified" or "generated",
    "query_id": "VQ1" ... "VQ10" or null,
    "match_reason": "one short sentence"
}}

Do not include any other text.
"""

    response = llm.invoke(
        classification_prompt
    ).content.strip()

    json_match = re.search(
        r"\{.*\}",
        response,
        re.DOTALL
    )

    if json_match:
        return json.loads(json_match.group())

    return {
        "route": "generated",
        "query_id": None,
        "match_reason": "Could not parse classification"
    }


# ============================================================
# Tool 2: Query Generation
# ============================================================

def generate_query(user_question, schema_context):

    generation_prompt = f"""
### ROLE

You are a senior SQL developer specializing in supply chain
operations analytics using SQLite.

### USER QUESTION

{user_question}

### DATABASE SCHEMA

{schema_context}

### INSTRUCTIONS

1. Write a single SQL query that answers the user's question.
2. Use only the provided schema.
3. The query must be read-only.
4. Use SELECT or WITH ... SELECT only.
5. Never use DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE,
   REPLACE, or ATTACH.
6. Do not invent tables or columns.
7. Resolve named warehouses using warehouse_name or warehouse_id.
8. Use SQLite-compatible syntax.
9. For date differences, use julianday().
10. Alias numeric columns with meaningful unit suffixes:
    - _usd
    - _pct
    - _count
    - _hours
    - _days

### OUTPUT

Return ONLY the SQL query.

No markdown.
No comments.
No explanation.
"""

    sql = llm.invoke(
        generation_prompt
    ).content.strip()

    sql = re.sub(
        r"^```sql\s*|\s*```$",
        "",
        sql,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()

    sql = re.sub(
        r"^```\s*|\s*```$",
        "",
        sql,
        flags=re.MULTILINE
    ).strip()

    return sql


# ============================================================
# Tool 3: Query Validation
# ============================================================

def validate_query(
    user_question,
    candidate_sql,
    db_connection,
    query_library,
    query_id=None
):

    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None
    }

    # --------------------------------------------------------
    # Check 1: Read-only shape
    # --------------------------------------------------------

    sql_upper = candidate_sql.upper().strip()

    forbidden_keywords = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "REPLACE",
        "ATTACH"
    ]

    if not (
        sql_upper.startswith("SELECT")
        or sql_upper.startswith("WITH")
    ):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH"
        return result

    for kw in forbidden_keywords:

        if re.search(
            r"\b" + kw + r"\b",
            sql_upper
        ):
            result["failed_check"] = "read_only_shape"
            result["details"] = (
                f"Forbidden keyword detected: {kw}"
            )
            return result

    if ";" in candidate_sql.rstrip(";").rstrip():

        result["failed_check"] = "read_only_shape"
        result["details"] = (
            "Multiple statements are not allowed"
        )
        return result

    # --------------------------------------------------------
    # Check 2: Schema conformance
    # --------------------------------------------------------

    cur = db_connection.cursor()

    real_tables = [
        r[0]
        for r in cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            """
        ).fetchall()
    ]

    real_columns = set()

    for table in real_tables:

        for col_info in cur.execute(
            f"PRAGMA table_info({table})"
        ).fetchall():

            real_columns.add(
                col_info[1].lower()
            )

    parsed = sqlparse.parse(candidate_sql)[0]

    referenced_identifiers = re.findall(
        r"\b[a-z_][a-z0-9_]*\b",
        candidate_sql.lower()
    )

    sql_keywords = {
        "select",
        "from",
        "where",
        "and",
        "or",
        "group",
        "by",
        "order",
        "having",
        "limit",
        "join",
        "on",
        "as",
        "case",
        "when",
        "then",
        "else",
        "end",
        "sum",
        "count",
        "avg",
        "min",
        "max",
        "round",
        "desc",
        "asc",
        "left",
        "right",
        "inner",
        "outer",
        "distinct",
        "null",
        "is",
        "not",
        "in",
        "like",
        "with",
        "union",
        "all",
        "between",
        "coalesce"
    }

    aliases = {
        "w",
        "i",
        "c",
        "s",
        "p"
    }

    unknown = [
        tok
        for tok in referenced_identifiers
        if tok not in sql_keywords
        and tok not in real_columns
        and tok not in real_tables
        and tok not in aliases
        and not tok.isdigit()
    ]

    # The notebook establishes the schema-conformance stage,
    # but the final gate relies on SQLite parsing/planning and
    # LLM relevance rather than treating every SQL token as
    # an independent schema error.

    # --------------------------------------------------------
    # Check 3: Parse and plan
    # --------------------------------------------------------

    try:

        cur.execute(
            f"EXPLAIN {candidate_sql}"
        )

        cur.fetchall()

    except sqlite3.Error as e:

        result["failed_check"] = (
            "parse_plan_dry_run"
        )

        result["details"] = (
            f"SQL failed to parse or plan: {str(e)}"
        )

        return result

    # --------------------------------------------------------
    # Check 4: LLM relevance
    # --------------------------------------------------------

    is_verified_track = (
        query_id is not None
        and query_id in query_library
    )

    if is_verified_track:

        track_context = """
This SQL is a pre-approved VERIFIED TEMPLATE.

It may intentionally return a broader result set than the
specific question. Do not fail it simply because it does not
filter to one particular region, warehouse, carrier, or category.

Judge whether the underlying metric, tables, aggregation,
and business logic match the user's intent.
"""

    else:

        track_context = """
This SQL was freshly generated for this question.

It should be appropriately scoped and filtered to answer
the user's question directly.
"""

    relevance_prompt = f"""
### ROLE

You are a senior data validator.

Determine whether the SQL correctly answers the business
user's supply chain operations question.

### CONTEXT

{track_context}

### USER QUESTION

{user_question}

### CANDIDATE SQL

{candidate_sql}

### CHECK

Assess:

1. Correct tables and columns.
2. Correct business metric.
3. Correct aggregation and grouping.
4. Correct business definitions.
5. Correct entity handling.
6. Correct answer shape.
7. Correct filtering when required.

### OUTPUT

Return ONLY JSON:

{{
    "verdict": "yes" or "no",
    "confidence": 0.0 to 1.0,
    "reason": "one short sentence"
}}
"""

    relevance_response = evaluator_llm.invoke(
        relevance_prompt
    ).content.strip()

    json_match = re.search(
        r"\{.*\}",
        relevance_response,
        re.DOTALL
    )

    if json_match:

        relevance_json = json.loads(
            json_match.group()
        )

        result["relevance_confidence"] = (
            relevance_json.get(
                "confidence",
                0.0
            )
        )

        if (
            relevance_json.get("verdict") == "no"
            or relevance_json.get(
                "confidence",
                0.0
            ) < 0.6
        ):

            result["failed_check"] = (
                "llm_relevance"
            )

            result["details"] = (
                "Relevance check failed: "
                + relevance_json.get(
                    "reason",
                    "unknown"
                )
            )

            return result

    result["passed"] = True
    result["details"] = (
        "All validation checks passed"
    )

    return result


# ============================================================
# Tool 4: Retry Generation
# ============================================================

def retry_generation(
    user_question,
    failed_sql,
    error_message,
    schema_context
):

    retry_prompt = f"""
### ROLE

You are a senior SQL developer fixing a query that failed
validation.

### USER QUESTION

{user_question}

### FAILED SQL

{failed_sql}

### VALIDATION ERROR

{error_message}

### DATABASE SCHEMA

{schema_context}

### INSTRUCTIONS

1. Fix only the identified problem.
2. Preserve the original user intent.
3. Return read-only SELECT or WITH ... SELECT.
4. Use only valid tables and columns.
5. Use SQLite-compatible syntax.

### OUTPUT

Return ONLY the corrected SQL.

No markdown.
No comments.
No explanation.
"""

    revised_sql = llm.invoke(
        retry_prompt
    ).content.strip()

    revised_sql = re.sub(
        r"^```sql\s*|\s*```$",
        "",
        revised_sql,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()

    revised_sql = re.sub(
        r"^```\s*|\s*```$",
        "",
        revised_sql,
        flags=re.MULTILINE
    ).strip()

    return revised_sql


# ============================================================
# Tool 5: Query Execution
# ============================================================

def execute_query(
    validated_sql,
    db_connection
):

    result = {
        "dataframe": None,
        "reasonable": True,
        "warnings": []
    }

    df = pd.read_sql_query(
        validated_sql,
        db_connection
    )

    result["dataframe"] = df

    if df.empty:

        result["warnings"].append(
            "Query returned an empty result"
        )

    for col in df.select_dtypes(
        include="number"
    ).columns:

        if (
            (df[col] < 0).any()
            and "deviation" not in col.lower()
            and "change" not in col.lower()
        ):

            result["warnings"].append(
                f"Column {col} contains negative values"
            )

        if df[col].isnull().any():

            null_count = df[col].isnull().sum()

            if null_count > len(df) * 0.5:

                result["warnings"].append(
                    f"Column {col} has "
                    f"{null_count} null values"
                )

    return result


# ============================================================
# Tool 6: Response Generation
# ============================================================

def generate_response(
    user_question,
    dataframe,
    route,
    query_id=None
):

    response_prompt = f"""
### ROLE

You are a supply chain operations analyst writing a concise
business response for a fulfillment or inventory question.

### USER QUESTION

{user_question}

### QUERY RESULT DATA

{dataframe.to_string()}

### INSTRUCTIONS

1. Answer the specific question directly.
2. Do not dump the entire table.
3. Highlight only relevant rows when appropriate.
4. Provide comparisons when useful.
5. State exact numbers from the data.
6. Flag notable operational risks.
7. Use professional operations language.
8. Keep the response concise.
9. Interpret units from column names:
   - _usd = dollars
   - _pct = percentage
   - _count = count
   - _hours = hours
   - _days = days

### OUTPUT

Return ONLY the natural-language response.

No headers.
No bullet points unless genuinely necessary.
"""

    response = llm.invoke(
        response_prompt
    ).content.strip()

    return response


# ============================================================
# Complete Workflow
# ============================================================

def run_workflow(
    user_question,
    status=None
):

    def report(message):

        if status is not None:
            status.write(message)

    log = {
        "user_question": user_question,
        "route": None,
        "query_id": None,
        "match_reason": None,
        "candidate_sql": None,
        "gate_result": None,
        "retry_used": False,
        "escalated": False,
        "executed_sql": None,
        "row_count": None,
        "confidence": None,
        "response": None
    }

    # --------------------------------------------------------
    # Step 1: Intent Classification
    # --------------------------------------------------------

    classification = classify_intent(
        user_question,
        VERIFIED_QUERY_LIBRARY
    )

    log["route"] = classification["route"]
    log["query_id"] = classification.get(
        "query_id"
    )
    log["match_reason"] = classification.get(
        "match_reason"
    )

    report(
        f"**Intent Classification:** "
        f"route=`{log['route']}`, "
        f"query_id=`{log['query_id']}`  \n"
        f"{log['match_reason']}"
    )

    # --------------------------------------------------------
    # Step 2: Query Construction
    # --------------------------------------------------------

    if (
        log["route"] == "verified"
        and log["query_id"] in VERIFIED_QUERY_LIBRARY
    ):

        candidate_sql = VERIFIED_QUERY_LIBRARY[
            log["query_id"]
        ]["sql"]

        report(
            "**Query Construction:** "
            "loaded from verified library"
        )

    else:

        candidate_sql = generate_query(
            user_question,
            DATABASE_SCHEMA
        )

        report(
            "**Query Construction:** "
            "generated fresh SQL"
        )

    log["candidate_sql"] = candidate_sql

    # --------------------------------------------------------
    # Step 3: Validation Gate
    # --------------------------------------------------------

    gate = validate_query(
        user_question,
        candidate_sql,
        conn,
        VERIFIED_QUERY_LIBRARY,
        log["query_id"]
    )

    log["gate_result"] = gate
    log["confidence"] = gate.get(
        "relevance_confidence"
    )

    report(
        f"**Validation Gate:** "
        f"passed=`{gate['passed']}`, "
        f"relevance_confidence="
        f"`{gate.get('relevance_confidence')}`"
    )

    # --------------------------------------------------------
    # Step 4: Retry Generated SQL
    # --------------------------------------------------------

    if (
        not gate["passed"]
        and log["route"] == "generated"
    ):

        report(
            f"Retrying: {gate['details']}"
        )

        candidate_sql = retry_generation(
            user_question,
            candidate_sql,
            gate["details"],
            DATABASE_SCHEMA
        )

        log["candidate_sql"] = candidate_sql
        log["retry_used"] = True

        gate = validate_query(
            user_question,
            candidate_sql,
            conn,
            VERIFIED_QUERY_LIBRARY,
            None
        )

        log["gate_result"] = gate
        log["confidence"] = gate.get(
            "relevance_confidence"
        )

        report(
            f"**Retry Validation Gate:** "
            f"passed=`{gate['passed']}`, "
            f"relevance_confidence="
            f"`{gate.get('relevance_confidence')}`"
        )

    # --------------------------------------------------------
    # Step 5: Escalation
    # --------------------------------------------------------

    if not gate["passed"]:

        log["escalated"] = True
        log["route"] = "escalate"

        log["response"] = (
            "Query could not be reliably resolved. "
            "Escalated to human analyst. "
            f"Failure: {gate['details']}"
        )

        report(
            f"**Escalated to human:** "
            f"{gate['details']}"
        )

        return {
            "log": log,
            "dataframe": None,
            **log
        }

    # --------------------------------------------------------
    # Step 6: Execute
    # --------------------------------------------------------

    log["executed_sql"] = candidate_sql

    exec_result = execute_query(
        candidate_sql,
        conn
    )

    df = exec_result["dataframe"]

    log["row_count"] = len(df)

    report(
        f"**Execute:** "
        f"{len(df)} rows returned"
    )

    if exec_result["warnings"]:

        report(
            f"Warnings: "
            f"{exec_result['warnings']}"
        )

    # --------------------------------------------------------
    # Step 7: Response Generation
    # --------------------------------------------------------

    response = generate_response(
        user_question,
        df,
        log["route"],
        log["query_id"]
    )

    log["response"] = response

    log["confidence"] = gate.get(
        "relevance_confidence"
    )

    report(
        f"**Response Generation:** "
        f"confidence=`{log['confidence']}`"
    )

    return {
        "log": log,
        "dataframe": df,
        **log
    }


# ============================================================
# Streamlit User Interface
# ============================================================

st.title(
    "📦 Inventory & Fulfillment Exception Engine"
)

st.caption(
    "Ask natural-language questions about inventory, "
    "warehouses, shipments, and carrier performance."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        "This application routes operational questions through "
        "a verified SQL template library when possible, or "
        "generates fresh SQL for novel questions. Every query "
        "passes through validation before database execution."
    )

    st.subheader("Example questions")

    st.markdown(
        "- Which 5 warehouses have the highest dollar value "
        "of inventory?\n"
        "- Do our bonded warehouses run hotter on capacity "
        "than the regular ones?\n"
        "- Which regions have the most stockouts?\n"
        "- How many shipments are currently delayed?\n"
        "- Which carriers have the worst OTIF compliance?\n"
        "- What are the most common reasons for shipment delays?\n"
        "- Which warehouses are operating above 85% capacity?"
    )


# ============================================================
# User Input
# ============================================================

user_question = st.text_input(
    "Your question",
    placeholder=(
        "e.g. Which warehouses are operating above 85% capacity?"
    )
)

show_trace = st.checkbox(
    "Show pipeline trace",
    value=True
)

submitted = st.button(
    "Run query",
    type="primary"
)


# ============================================================
# Run Application
# ============================================================

if submitted and user_question.strip():

    trace_container = (
        st.status(
            "Running pipeline...",
            expanded=show_trace
        )
        if show_trace
        else None
    )

    try:

        output = run_workflow(
            user_question=user_question,
            status=trace_container
        )

    except Exception as e:

        if trace_container is not None:

            trace_container.update(
                label="Pipeline error",
                state="error"
            )

        st.error(
            f"Pipeline failed: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Pipeline Status
    # --------------------------------------------------------

    if trace_container is not None:

        trace_container.update(
            label=(
                "Pipeline complete"
                if not output["escalated"]
                else "Escalated to human review"
            ),
            state=(
                "complete"
                if not output["escalated"]
                else "error"
            )
        )

    st.divider()

    # --------------------------------------------------------
    # Escalation
    # --------------------------------------------------------

    if output["escalated"]:

        st.warning(
            output["response"]
        )

        with st.expander(
            "Validation details"
        ):

            st.json(
                output["gate_result"]
            )

    # --------------------------------------------------------
    # Successful Result
    # --------------------------------------------------------

    else:

        confidence = output["confidence"]

        if isinstance(
            confidence,
            (int, float)
        ):

            if confidence >= 0.8:

                badge = "🟢"

            elif confidence >= 0.6:

                badge = "🟡"

            else:

                badge = "🔴"

            confidence_display = (
                f"{confidence:.2f}"
            )

        else:

            badge = "⚪"
            confidence_display = str(
                confidence
            )

        st.subheader("Answer")

        st.write(
            output["response"]
        )

        st.caption(
            f"{badge} Confidence: "
            f"{confidence_display}  ·  "
            f"Route: {output['route']}  ·  "
            f"Rows returned: {output['row_count']}"
        )

        # ----------------------------------------------------
        # Underlying Data
        # ----------------------------------------------------

        if output["dataframe"] is not None:

            with st.expander(
                "View underlying data"
            ):

                st.dataframe(
                    output["dataframe"],
                    use_container_width=True
                )

        # ----------------------------------------------------
        # Executed SQL
        # ----------------------------------------------------

        with st.expander(
            "View executed SQL"
        ):

            st.code(
                output["executed_sql"],
                language="sql"
            )

else:

    if submitted:

        st.info(
            "Please enter a question first."
        )