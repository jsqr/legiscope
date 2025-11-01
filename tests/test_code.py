from legiscope.code import split_paragraph, match_heading
from legiscope.code import HeadingPattern, Heading
from legiscope.code import chunkify_paragraph, chunkify_paragraphs


def test_split_paragraph():
    assert split_paragraph("") == ("", "")
    assert split_paragraph("This is a\nparagraph.\n") == ("This is a", "paragraph.\n")


def test_match_heading():
    test_doc1 = "Chapter VII: The Final Chapter"
    test_pattern1 = HeadingPattern(
        level=1, regex=r"^Chapter ([IVXLC]+): (.+)$", multi_line=False
    )

    test_doc2 = "\n\nChapter 7:\nThe Final Chapter"
    test_pattern2 = HeadingPattern(level=1, regex=r"^Chapter (\d+):", multi_line=True)

    assert match_heading(test_doc1, {1: test_pattern1}) == Heading(
        level=1, enumeration="VII", heading_text="The Final Chapter"
    )

    assert match_heading(test_doc2, {1: test_pattern2}) == Heading(
        level=1, enumeration="7", heading_text="The Final Chapter"
    )


nyc_example_headings = {
    1: [
        "Title 1: General Provisions\n",
        "Title 2: City of New York\n",
        "Title 3: Elected officials\n",
    ],
    2: [
        "Chapter 1: Powers and Rights of the Corporation; Emblems and Insignia\n",
        "Chapter 2: Boundaries of the City\n",
        "Chapter 4: Board of Estimate\n",
    ],
    3: [
        "§ 2-101 Name; powers and rights of the corporation; seal.\n",
        "§ 2-202 Division into boroughs and boundaries thereof.\n",
        "§ 3-140 Office of labor standards.\n",
    ],
}

chicago_example_headings = {
    1: [
        "TITLE 1\nGENERAL PROVISION\n",
        "TITLE 2\nCITY GOVERNMENT AND ADMINISTRATION\n",
        "TITLE 3\nREVENUE AND FINANCE\n",
    ],
    2: [
        "CHAPTER 1-4\nCODE ADOPTION - ORGANIZATION\n",
        "CHAPTER 1-8\nCITY SEAL AND FLAG\n",
        "CHAPTER 1-12\nCITY EMBLEMS\n",
    ],
    3: [
        "1-4-010 Municipal Code of Chicago adopted.\n",
        "2-1-020 Code to be kept up-to-date.\n",
        "3-4-030 Official copy on file.\n",
    ],
}

losangeles_example_headings = {
    1: [
        "CHAPTER I\nGENERAL PROVISIONS AND ZONING\n",
        "CHAPTER IV\nPUBLIC WELFARE",
        "CHAPTER VII\nTRANSPORTATION\n",
    ],
    2: [
        "ARTICLE 1\nGENERAL PROVISIONS\n",
        "ARTICLE 4\nPUBLIC BENEFIT PROJECTS\n",
        "ARTICLE 4.3\nELDERCARE FACILITY UNIFIED PERMIT PROCESS\n",
    ],
    3: [
        "SEC. 11.00. PROVISIONS APPLICABLE TO CODE.\n",
        "SEC. 11.01. DEFINITIONS AND INTERPRETATION.\n",
        "SEC. 14.4.1. PURPOSE.\n",
    ],
}

test_paragraph = (
    "This is a paragraph. It has multiple words. It has multiple sentences."
)
test_paragraphs = ["Paragraph one.", "Paragraph two.", "Paragraph three."]


def test_chunkify_paragraph():
    assert chunkify_paragraph(test_paragraph, 100) == [test_paragraph]
    assert chunkify_paragraph(test_paragraph, 8) == [
        "This is a paragraph. It has multiple words.",
        "It has multiple sentences.",
    ]
    return


def test_chunkify_paragraphs():
    assert chunkify_paragraphs(test_paragraphs, 100) == [
        "Paragraph one.\nParagraph two.\nParagraph three."
    ]
    assert chunkify_paragraphs(test_paragraphs, 4) == [
        "Paragraph one.\nParagraph two.",
        "Paragraph three.",
    ]
    return
