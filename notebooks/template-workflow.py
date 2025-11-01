import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import sys
    sys.path.insert(0, '..')
    return mo, sys


@app.cell
def __(mo):
    mo.md(
        """
        ## Getting ready
        
        Begin by:
        1. creating a new directory `data/<jurisdiction>` and populate with one or more
        docx files containing the jurisdiction's municipal code
        2. run `scripts/convert_docx.sh` to convert those files into a single text file
        3. make a copy of this notebook to `notebooks/<jurisdiction>.py` 
        and continue processing in that notebook
        """
    )
    return


@app.cell
def __(mo):
    # Install dependencies
    mo.run(
        [
            "%pip install -qU openai marvin",
            "%pip install -qU \"psycopg[binary]\""
        ]
    )
    return


@app.cell
def __():
    from legiscope.code import Jurisdiction
    return Jurisdiction,


@app.cell
def __(mo):
    mo.md("## Specify heading patterns")
    return


@app.cell
def __(mo):
    mo.md("Replace the `jurisdiction_headings` dict with examples from your jurisdiction")
    return


@app.cell
def __():
    heading_examples = {
        1: ["TITLE 1\nGENERAL PROVISION\n",
            "TITLE 2\nCITY GOVERNMENT AND ADMINISTRATION\n",
            "TITLE 3\nREVENUE AND FINANCE\n",
            ],
        2: ["CHAPTER 1-4\nCODE ADOPTION - ORGANIZATION\n",
            "CHAPTER 1-8\nCITY SEAL AND FLAG\n",
            "CHAPTER 1-12\nCITY EMBLEMS\n",
            ],
        3: ["1-4-010 Municipal Code of Chicago adopted.\n",
            "2-1-020 Code to be kept up-to-date.\n",
            "3-4-030 Official copy on file.\n",
            ],
    }
    return heading_examples,


@app.cell
def __():
    from legiscope.code import infer_heading_patterns, infer_level_names
    return infer_heading_patterns, infer_level_names,


@app.cell
def __(heading_examples, infer_heading_patterns, infer_level_names):
    # Verify that the regular expressions matching outline levels look okay
    heading_patterns = infer_heading_patterns(heading_examples)
    
    pattern_output = []
    for level, pattern in heading_patterns.items():
        pattern_output.append(f"{level}: r'{pattern.regex}'")
    
    # Verify that the names of the sections look okay
    level_names = infer_level_names(heading_patterns)
    
    name_output = []
    for level, name in level_names.items():
        name_output.append(f"{level}: {name}")
    return heading_patterns, level_names, name_output, pattern_output


@app.cell
def __(mo, name_output, pattern_output):
    mo.md("### Heading Patterns")
    mo.md("\n".join(pattern_output))
    
    mo.md("### Level Names")
    mo.md("\n".join(name_output))
    return


@app.cell
def __(mo):
    mo.md("## Specify the parameters of the jurisdiction and parse the code")
    return


@app.cell
def __(Jurisdiction, heading_patterns, level_names):
    place = Jurisdiction(
        name="Chicago Mini",
        title="Municipal Code of Chicago",
        patterns=heading_patterns,
        level_names=level_names,
        source_local="../data/chicago-mini/code.txt",
        source_url="https://www.chicago.gov/city/en/depts/doit/supp_info/municipal_code.html",
    )
    
    place.parse()
    place.chunkify(1000)
    return place,


@app.cell
def __(mo, place):
    # Verify that the distribution of paragraphs and chunks looks okay
    summary_output = place.summarize()
    mo.md(f"### Summary\n\n```\n{summary_output}\n```")
    return


@app.cell
def __(mo):
    mo.md("## Upload data to the database")
    return


@app.cell
def __():
    from legiscope.code import upload
    
    db = {'dbname': 'muni',
          'user': 'muni',
          'password': '',
          'host': 'localhost',
          'port': 5432}
    return db, upload


@app.cell
def __(db, place, upload):
    upload(db, place)
    return


@app.cell
def __():
    from legiscope.code import upload_embeddings, refresh_views
    return refresh_views, upload_embeddings,


@app.cell
def __(db, place, refresh_views, upload_embeddings):
    upload_embeddings(db, place)
    refresh_views(db)
    return


@app.cell
def __(mo):
    mo.md("## Find associations among sections")
    return


@app.cell
def __():
    from legiscope.code import find_associations
    return find_associations,


@app.cell
def __(db, find_associations, place):
    find_associations(db, place)
    # TODO: changing DB schema
    return


@app.cell
def __(mo):
    mo.md("## Basic queries")
    return


@app.cell
def __():
    from legiscope.code import simple_semantic_query
    return simple_semantic_query,


@app.cell
def __(mo, simple_semantic_query):
    # Semantic query interface
    semantic_query = mo.ui.text_area(
        label="Semantic Query",
        placeholder="Enter your question about the municipal code...",
        value="Does the municipal code contain provisions restricting the use of drug paraphernalia?"
    )
    
    semantic_limit = mo.ui.slider(
        label="Result Limit",
        start=5,
        stop=50,
        step=5,
        value=20
    )
    
    return semantic_limit, semantic_query


@app.cell
def __(db, place, semantic_limit, semantic_query, simple_semantic_query):
    def run_semantic_query():
        if semantic_query.value:
            results = simple_semantic_query(db, place, semantic_query.value, limit=semantic_limit.value)
            return results
        return []
    
    semantic_results = run_semantic_query()
    return run_semantic_query, semantic_results


@app.cell
def __(mo, semantic_query, semantic_results):
    mo.md(f"### Query: {semantic_query.value}")
    
    if semantic_results:
        result_text = "\n\n".join([str(result) for result in semantic_results])
        mo.md(f"### Results\n\n{result_text}")
    else:
        mo.md("No results to display. Enter a query above.")
    return


@app.cell
def __():
    from legiscope.code import extract_keywords, simple_full_text_query
    return extract_keywords, simple_full_text_query,


@app.cell
def __(mo, simple_full_text_query):
    # Full-text query interface
    fulltext_query = mo.ui.text_area(
        label="Full-Text Query",
        placeholder="Enter keywords for full-text search...",
        value="drug paraphernalia"
    )
    
    fulltext_limit = mo.ui.slider(
        label="Result Limit",
        start=5,
        stop=50,
        step=5,
        value=20
    )
    
    return fulltext_limit, fulltext_query


@app.cell
def __(db, fulltext_limit, fulltext_query, place, simple_full_text_query):
    def run_fulltext_query():
        if fulltext_query.value:
            results = simple_full_text_query(db, place, fulltext_query.value, limit=fulltext_limit.value)
            return results
        return []
    
    fulltext_results = run_fulltext_query()
    return run_fulltext_query, fulltext_results


@app.cell
def __(fulltext_query, fulltext_results, mo):
    mo.md(f"### Query: {fulltext_query.value}")
    
    if fulltext_results:
        result_text = "\n\n".join([str(result) for result in fulltext_results])
        mo.md(f"### Results\n\n{result_text}")
    else:
        mo.md("No results to display. Enter a query above.")
    return


@app.cell
def __():
    from legiscope.code import hybrid_query
    return hybrid_query,


@app.cell
def __(hybrid_query, mo):
    # Hybrid query interface
    hybrid_query_input = mo.ui.text_area(
        label="Hybrid Query",
        placeholder="Enter your question for hybrid semantic + full-text search...",
        value="Does the municipal code contain provisions restricting the use of drug paraphernalia?"
    )
    
    hybrid_limit = mo.ui.slider(
        label="Result Limit",
        start=5,
        stop=50,
        step=5,
        value=20
    )
    
    return hybrid_limit, hybrid_query_input


@app.cell
def __(db, hybrid_limit, hybrid_query_input, hybrid_query, place):
    def run_hybrid_query():
        if hybrid_query_input.value:
            results = hybrid_query(db, place, hybrid_query_input.value, limit=hybrid_limit.value)
            return results
        return []
    
    hybrid_results = run_hybrid_query()
    return hybrid_results, run_hybrid_query


@app.cell
def __(hybrid_query_input, hybrid_results, mo):
    mo.md(f"### Query: {hybrid_query_input.value}")
    
    if hybrid_results:
        result_text = "\n\n".join([str(result) for result in hybrid_results])
        mo.md(f"### Results\n\n{result_text}")
    else:
        mo.md("No results to display. Enter a query above.")
    return


@app.cell
def __(mo):
    mo.md("## Report generation")
    return


@app.cell
def __():
    from legiscope.code import Report
    return Report,


@app.cell
def __(mo, Report):
    # Report generation interface
    report_query = mo.ui.text_area(
        label="Report Query",
        placeholder="Enter question for report generation...",
        value="Does the municipal code contain provisions restricting the use of drug paraphernalia?"
    )
    
    generate_report_btn = mo.ui.button(label="Generate Report")
    
    return generate_report_btn, report_query


@app.cell
def __(Report, db, generate_report_btn, place, report_query):
    def generate_report():
        if generate_report_btn.value and report_query.value:
            report = Report(db, place, report_query.value)
            return str(report)
        return ""
    
    report_content = generate_report()
    return generate_report, report_content


@app.cell
def __(mo, report_content, report_query):
    mo.md(f"### Report Query: {report_query.value}")
    
    if report_content:
        mo.md(report_content)
    else:
        mo.md("Click 'Generate Report' to create a report based on your query.")
    return


@app.cell
def __(mo):
    mo.md("## Upload results to database")
    return


@app.cell
def __():
    from legiscope.code import upload_report
    return upload_report,


@app.cell
def __(Report, db, mo, place, report_query, upload_report):
    # Upload report interface
    upload_report_btn = mo.ui.button(label="Upload Report to Database")
    
    def upload_report_handler():
        if upload_report_btn.value and report_query.value:
            report = Report(db, place, report_query.value)
            upload_report(db, report)
            return "Report uploaded successfully!"
        return "Enter a query and click 'Generate Report' first."
    
    upload_status = upload_report_handler()
    
    mo.md(f"### Upload Status: {upload_status}")
    return upload_report_btn, upload_report_handler, upload_status


if __name__ == "__main__":
    app.run()