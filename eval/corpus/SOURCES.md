# Evaluation corpus sources

All documents are U.S. government works (CFPB), public domain, no license question.

| file | origin URL | sha256 | document type | pages | scanned |
|---|---|---|---|---|---|
| cfpb_closing_disclosure_h25a_model.pdf | https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25A.pdf | 2d80d1746fa50a6292da1367a5367e80f1637f16c8d839509c80d833c0d027fe | Closing Disclosure (H-25(A) model form) | 14 | no |
| cfpb_closing_disclosure_h25b.pdf | https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25B.pdf | 606a93c8baaca815439822df5cf8c78cbb2dcf6cc4af5aa291a459c7917e4173 | Closing Disclosure (H-25(B) fixed-rate loan sample) | 6 | no |
| cfpb_closing_disclosure_h25c_second_lien.pdf | https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25C.pdf | 67b049ebf8b77f2b19d7ad8af03dff0ff27d86fd278f77c5f209d1cd6105f2ee | Closing Disclosure (H-25(C) second-lien sample) | 2 | no |
| cfpb_closing_disclosure_h25d_seller_second_lien.pdf | https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25D1.pdf | cc857e50d13af7ea736503bf1ca6ffbe1901c1c7a530d327a02773884825251e | Closing Disclosure (H-25(D) borrower satisfaction of seller second-lien sample) | 2 | no |
| cfpb_closing_disclosure_h25e_refinance.pdf | https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25E.pdf | 6c05ffba10741d55d5bdc6dd946eefc0873fd6088a19c3a24219fb2ed8aa341a | Closing Disclosure (H-25(E) refinance sample, matches LE H-24(D)) | 6 | no |
| cfpb_closing_disclosure_h25g.pdf | https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25G.pdf | 06386504873d5d3d8b2268a128a1f9f39d5fcd6407a79ca1d611a9fa94c3789c | Closing Disclosure (H-25(G) sample) | 6 | no |
| cfpb_closing_disclosure_h25h.pdf | https://files.consumerfinance.gov/f/201403_cfpb_closing-disclosure_cover-H25H.pdf | e9917229b06d2f55a826909022be4a366d0c44b1e627790443c981d331119c7d | Closing Disclosure (H-25(H) sample) | 3 | no |
| cfpb_closing_forms_guide.pdf | https://files.consumerfinance.gov/f/documents/cfpb_buying-a-house_closing-forms_guide.pdf | 8e0785f087fba9f42f05c2d421f9cda0db1dcdea5f07295a965d1348be66855a | Consumer guide (closing forms) | 5 | no |
| cfpb_home_loan_toolkit.pdf | https://files.consumerfinance.gov/f/documents/cfpb_your-home-loan-toolkit.pdf | 7f9c2663177950356fd2960f0ca55466530d7eacf1ec96750ee2930840625229 | Consumer guide (Your Home Loan Toolkit booklet) | 28 | no |
| cfpb_loan_estimate_h24b_fixed_rate.pdf | https://files.consumerfinance.gov/f/201403_cfpb_loan-estimate_fixed-rate-loan-sample-H24B.pdf | 243551dbce6362e616328924eaf5b1818b734883d43ec91a73c160e5da52b385 | Loan Estimate (H-24(B) fixed-rate loan sample) | 4 | no |
| cfpb_loan_estimate_h24c_interest_only_arm.pdf | https://files.consumerfinance.gov/f/201403_cfpb_loan-estimate_interest-only-adjustable-rate-loan-sample-H24C.pdf | a74396afac2a929ab33c12e14ecfa42984be9544250a93dd7b2284778cabd791 | Loan Estimate (H-24(C) interest-only ARM sample) | 4 | no |
| cfpb_loan_estimate_h24d_refinance.pdf | https://files.consumerfinance.gov/f/201403_cfpb_loan-estimate_refinance-sample-H24D.pdf | baadbe3d1f1254f422c6ab30b53dac91649be9d5300702cd5068bc1090be1560 | Loan Estimate (H-24(D) refinance sample) | 4 | no |
| cfpb_loan_estimate_h24e_balloon.pdf | https://files.consumerfinance.gov/f/201403_cfpb_loan-estimate_baloon-payment-H24E.pdf | 8659f8a3a32fdffcabc48be902f596fee25302c648910331d482db41d3d80d4d | Loan Estimate (H-24(E) balloon payment sample) | 2 | no |
| cfpb_loan_estimate_h24f_negative_amortization.pdf | https://files.consumerfinance.gov/f/201403_cfpb_loan-estimate_negative-amortization-sample-H24F.pdf | bdbe3a5be304abb27ed85af89d6104ecbb42538f62d8d58887d9a40608033307 | Loan Estimate (H-24(F) negative amortization sample) | 2 | no |
| cfpb_mortgage_closing_checklist_scanned.pdf | https://files.consumerfinance.gov/f/documents/cfpb_buying-a-house_mortgage-closing_checklist.pdf | 86a68263de0754231e0a537b40557e40c7e1d392ca92de932c59c87c0dba10c0 | Consumer guide (mortgage closing checklist) | 6 | yes (rasterized from https://files.consumerfinance.gov/f/documents/cfpb_buying-a-house_mortgage-closing_checklist.pdf) |

Notes:
- `cfpb_loan_estimate_h24b_fixed_rate.pdf` and `cfpb_closing_disclosure_h25b.pdf` are CFPB's matched Loan Estimate / Closing Disclosure pair for the same fixed-rate sample loan, used for cross-document `multi_hop` questions.
- `cfpb_closing_disclosure_h25b.pdf` is copied (not moved) from `tests/fixtures/cfpb_closing_disclosure.pdf`.
- `cfpb_mortgage_closing_checklist_scanned.pdf` was rasterized page-by-page at 150 dpi with `pypdfium2` (`page.render(scale=150/72).to_pil()`) and saved as an image-only PDF via `PIL.Image.save(..., save_all=True)`; the text original was removed from the corpus. It has no text layer (`get_textpage().get_text_range()` is empty on every page).
