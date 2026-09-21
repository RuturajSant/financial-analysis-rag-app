"""
Sample financial data generator.
Produces a realistic synthetic PDF annual report with:
  - A narrative business overview section
  - An income statement table
  - A balance sheet table
  - A cash flow statement table
  - Footnotes

Run: python data/generate_sample.py
Output: data/sample_annual_report.pdf
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT_PATH = Path(__file__).parent / "sample_annual_report.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, spaceAfter=6)


def _table(data: list[list], col_widths: list[float] | None = None) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ECF0F1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ECF0F1")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build_pdf() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )
    story = []

    # Cover / title
    story.append(Paragraph("Acme Corp Annual Report — Fiscal Year 2024", H1))
    story.append(Spacer(1, 12))

    # ── Page 1: Business Overview ──────────────────────────────────────────────
    story.append(Paragraph("1. Business Overview", H2))
    story.append(
        Paragraph(
            "Acme Corp (NASDAQ: ACME) is a diversified technology company focused on enterprise "
            "software solutions and cloud infrastructure services. Founded in 2005 and headquartered "
            "in San Francisco, California, Acme serves over 12,000 business customers across 45 countries.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "In fiscal year 2024 (ending December 31, 2024), Acme delivered strong revenue growth "
            "of 18.4% year-over-year, driven by expansion in its SaaS segment and continued adoption "
            "of its cloud infrastructure platform. Net income increased by 24.7% reflecting operating "
            "leverage and disciplined cost management.",
            BODY,
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Segment Highlights", H2))
    story.append(
        Paragraph(
            "The SaaS segment contributed $1,240 million in revenue (65% of total), while Cloud "
            "Infrastructure contributed $668 million (35% of total). Both segments saw double-digit "
            "growth compared to the prior fiscal year.",
            BODY,
        )
    )
    story.append(Spacer(1, 18))

    # ── Page 2: Income Statement ───────────────────────────────────────────────
    story.append(Paragraph("2. Consolidated Income Statement", H2))
    story.append(Paragraph("(In USD millions, for the year ended December 31)", BODY))
    story.append(Spacer(1, 6))

    income_data = [
        ["Line Item", "FY 2024", "FY 2023", "YoY Change"],
        ["Revenue", "$1,908", "$1,611", "+18.4%"],
        ["Cost of Goods Sold (COGS)", "$763", "$660", "+15.6%"],
        ["Gross Profit", "$1,145", "$951", "+20.4%"],
        ["Gross Margin %", "60.0%", "59.0%", "+100 bps"],
        ["Operating Expenses", "$612", "$540", "+13.3%"],
        ["Operating Income (EBIT)", "$533", "$411", "+29.7%"],
        ["Interest Expense", "$28", "$31", "-9.7%"],
        ["Pre-Tax Income", "$505", "$380", "+32.9%"],
        ["Income Tax Expense", "$101", "$76", "+32.9%"],
        ["Net Income", "$404", "$304", "+32.9%"],  # Note: simplified; tax rate 20%
        ["Earnings Per Share (EPS) — Basic", "$4.04", "$3.04", "+32.9%"],
        ["Weighted Avg. Shares Outstanding (M)", "100.0", "100.0", "—"],
    ]
    story.append(_table(income_data, col_widths=[2.5 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Note 1: Net income margin improved to 21.2% in FY 2024 from 18.9% in FY 2023, "
            "reflecting scale benefits in the SaaS segment and reduced cloud infrastructure unit costs.",
            BODY,
        )
    )
    story.append(Spacer(1, 18))

    # ── Page 3: Balance Sheet ──────────────────────────────────────────────────
    story.append(Paragraph("3. Consolidated Balance Sheet", H2))
    story.append(Paragraph("(In USD millions, as of December 31)", BODY))
    story.append(Spacer(1, 6))

    balance_data = [
        ["Item", "Dec 31, 2024", "Dec 31, 2023"],
        # Assets
        ["ASSETS", "", ""],
        ["Cash and Cash Equivalents", "$520", "$310"],
        ["Accounts Receivable, net", "$340", "$280"],
        ["Prepaid Expenses and Other", "$85", "$70"],
        ["Total Current Assets", "$945", "$660"],
        ["Property, Plant & Equipment, net", "$410", "$380"],
        ["Intangible Assets, net", "$250", "$290"],
        ["Goodwill", "$600", "$600"],
        ["Other Long-term Assets", "$95", "$70"],
        ["Total Assets", "$2,300", "$2,000"],
        # Liabilities
        ["LIABILITIES & EQUITY", "", ""],
        ["Accounts Payable", "$120", "$100"],
        ["Accrued Liabilities", "$180", "$160"],
        ["Deferred Revenue (current)", "$210", "$175"],
        ["Total Current Liabilities", "$510", "$435"],
        ["Long-term Debt", "$350", "$400"],
        ["Deferred Revenue (long-term)", "$90", "$75"],
        ["Other Long-term Liabilities", "$50", "$40"],
        ["Total Liabilities", "$1,000", "$950"],
        # Equity
        ["Common Stock & APIC", "$500", "$500"],
        ["Retained Earnings", "$800", "$550"],
        ["Total Stockholders' Equity", "$1,300", "$1,050"],
        ["Total Liabilities & Equity", "$2,300", "$2,000"],
    ]
    story.append(_table(balance_data, col_widths=[3.0 * inch, 1.5 * inch, 1.5 * inch]))
    story.append(Spacer(1, 18))

    # ── Page 4: Cash Flow ──────────────────────────────────────────────────────
    story.append(Paragraph("4. Consolidated Statement of Cash Flows", H2))
    story.append(Paragraph("(In USD millions, for the year ended December 31)", BODY))
    story.append(Spacer(1, 6))

    cashflow_data = [
        ["Activity", "FY 2024", "FY 2023"],
        ["Net Income", "$404", "$304"],
        ["Depreciation & Amortization", "$95", "$85"],
        ["Changes in Working Capital", "($45)", "($30)"],
        ["Cash from Operating Activities", "$454", "$359"],
        ["Capital Expenditures", "($75)", "($60)"],
        ["Acquisitions", "$0", "($50)"],
        ["Cash from Investing Activities", "($75)", "($110)"],
        ["Debt Repayment", "($50)", "($50)"],
        ["Share Repurchases", "($120)", "($80)"],
        ["Dividends Paid", "($0)", "($0)"],
        ["Cash from Financing Activities", "($170)", "($130)"],
        ["Net Change in Cash", "$209", "$119"],
        ["Beginning Cash", "$311", "$192"],  # slight rounding vs balance sheet for realism
        ["Ending Cash", "$520", "$311"],
    ]
    story.append(_table(cashflow_data, col_widths=[3.0 * inch, 1.5 * inch, 1.5 * inch]))
    story.append(Spacer(1, 12))

    # ── Footnotes ──────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Notes to Financial Statements", H2))
    story.append(
        Paragraph(
            "Note 2 — Revenue Recognition: Revenue from SaaS subscriptions is recognized ratably "
            "over the contract term. Professional services revenue is recognized as services are delivered. "
            "The company adopted ASC 606 effective January 1, 2019 with no material retrospective adjustments.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "Note 3 — Debt: The company's $350 million long-term debt consists of a revolving credit "
            "facility with a maturity date of June 30, 2027 at an interest rate of SOFR + 1.5%. "
            "The company repaid $50 million during FY 2024, reducing the balance from $400 million.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "Note 4 — Goodwill: Goodwill of $600 million relates entirely to the acquisition of "
            "DataSoft Inc. in fiscal year 2021. No impairment charges were recorded in FY 2024 "
            "or FY 2023. The company performs its annual goodwill impairment assessment in Q4.",
            BODY,
        )
    )

    doc.build(story)
    print(f"Sample PDF written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
