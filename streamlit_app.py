import streamlit as st
import pandas as pd
import openai
import json

# --- CONFIG ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🧠 AI Attribute Discovery Tool")
st.write("Upload a CSV file containing existing product attributes. This tool will analyze the data and suggest a rich set of standard attribute headers that can be used to guide LLM enrichment.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV of Existing Product Data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # --- Analyse existing columns ---
    sample_rows = df.head(10).to_dict(orient="records")
    columns = df.columns.tolist()

    st.subheader("Step 1: Enriching Attribute Schema via GPT")

    prompt = f"""
You are an expert product information architect.

Below is a sample of product data including available attribute columns. Your job is to suggest an extensive and normalized set of attribute headers that could be used to enrich these products using AI (e.g., GPT, image recognition, web lookup). These should include both common and advanced attributes—anything that can be reasonably inferred or standardized for high-quality product enrichment.

The output should be a JSON list of attribute names that would be useful to standardize and populate for these types of products. Be thoughtful and exhaustive.

Column Headers: {columns}

Sample Rows:
{sample_rows}

Suggested Attribute Headers:
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    output = response["choices"][0]["message"]["content"]
    
    st.subheader("💡 Suggested Rich Attribute Headers")
    st.code(output, language="json")

    # --- Editable JSON Editor ---
    st.subheader("✍️ Review and Edit Suggested Attributes")
    try:
        editable_list = st.text_area("Edit the list of attribute headers as needed (JSON format)", value=output, height=300)
        parsed_attributes = json.loads(editable_list)
        st.success("Attribute schema parsed successfully.")

        # Save schema if needed
        if st.button("💾 Save Attribute Schema"):
            with open("suggested_schema.json", "w") as f:
                json.dump(parsed_attributes, f, indent=2)
            st.success("Schema saved to suggested_schema.json")

        # --- Step 2: Apply Schema to Enrich Data ---
        st.subheader("✨ Step 2: Enrich Sample Products Using Suggested Schema")
        enriched_rows = []
        for row in sample_rows:
            product_text = json.dumps(row)
            enrichment_prompt = f"""
Given the following product data:

{product_text}

Populate as many of the following enriched attributes as possible:
{parsed_attributes}

Return the result as a JSON object.
"""

            enrich_response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": enrichment_prompt}],
                temperature=0.3
            )

            result = enrich_response["choices"][0]["message"]["content"]
            try:
                enriched_row = json.loads(result)
                enriched_rows.append(enriched_row)
            except:
                enriched_rows.append({"error": "Invalid JSON"})

        enriched_df = pd.DataFrame(enriched_rows)
        st.write("🔍 Enriched Sample Rows")
        st.dataframe(enriched_df)

        enriched_csv = enriched_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Enriched Sample CSV", enriched_csv, "enriched_sample.csv", "text/csv")

    except Exception as e:
        st.error(f"Error parsing attribute list: {e}")
