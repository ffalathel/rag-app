from docuchat.chunking import detect_table_content


def test_detects_currency_table():
    text = (
        "Loan Amount   $250,000.00   $1,234.56\n"
        "Taxes   $3,200.00   $266.67\n"
        "Insurance   $1,800.00   $150.00"
    )
    assert detect_table_content(text) is True


def test_prose_is_not_a_table():
    text = "Borrower shall pay to Lender the principal sum of two hundred fifty thousand dollars."
    assert detect_table_content(text) is False
