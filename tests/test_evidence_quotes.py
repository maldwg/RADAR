"""Protect scientific evidence while resolving PDF and model transcription differences."""
from __future__ import annotations

import unittest

from unittest.mock import patch

from radar.common.evidence_quotes import QUOTE_GROUNDING, SourceQuoteIndex
from radar.common.ollama import EvidenceSection, decision_profile, format_evidence, verify_evidence


class SourceQuotationTests(unittest.TestCase):
    """Exercise literal grounding, bounded case repair, ambiguity, and source suggestions."""

    def test_pdf_typography_returns_original_source_span(self):
        """Keep source bytes and offsets through whitespace, ligatures, and typographic quotes."""
        text = 'Preface. The “treatment” eﬀects\n are\t estimated. Appendix.'
        quote = 'The "treatment" effects are estimated.'
        match = SourceQuoteIndex(text).match(quote)
        self.assertIsNotNone(match)
        self.assertEqual(match.text, 'The “treatment” eﬀects\n are\t estimated.')
        self.assertEqual(text[match.start:match.end], match.text)

    def test_initial_article_capitalization_is_grounded_without_model_retry(self):
        """Accept the observed sentence-fragment capitalization mismatch using the actual source."""
        text = 'We observe that a major source of power loss is the data splitting step.'
        match = SourceQuoteIndex(text).match('A major source of power loss is the data splitting step.')
        self.assertEqual(match.text, 'a major source of power loss is the data splitting step.')
        self.assertEqual(match.method, 'initial_case')

    def test_case_repair_requires_an_unambiguous_span(self):
        """Avoid silently selecting among repeated, nonliteral capitalization matches."""
        self.assertIsNone(SourceQuoteIndex('the method works. Again, the method works.').match('The method works.'))
        text = 'data major source of power loss. a major source of power loss. a major source of power loss.'
        self.assertIsNone(SourceQuoteIndex(text).match('A major source of power loss.'))

    def test_changes_to_numbers_negation_and_notation_are_never_grounded(self):
        """Do not turn mathematical, numerical, or semantic edits into accepted quotations."""
        pairs = [
            ('X is independent of treatment.', 'x is independent of treatment.'),
            ('The method assumes X = 1.', 'The method assumes x = 1.'),
            ('The interval covers 75% of observations.', 'The interval covers 95% of observations.'),
            ('The method does not require overlap.', 'The method does require overlap.'),
            ('The method assumes α ≤ 0.05.', 'The method assumes α < 0.05.'),
            ('SUTVA is required.', 'sutva is required.'),
            ('The treatment effects are estimated.', 'The treatment effects are predicted.'),
        ]
        for text, quote in pairs:
            with self.subTest(quote=quote):
                self.assertIsNone(SourceQuoteIndex(text).match(quote))

    def test_spelling_corrections_are_not_accepted_matches(self):
        """Reject model autocorrections rather than turning rewording into accepted evidence."""
        text = 'Introduction. The calibartion procedure does not require constant variance. Discussion.'
        quote = 'The calibration procedure does not require constant variance.'
        index = SourceQuoteIndex(text)
        self.assertIsNone(index.match(quote))

    def test_expanding_ligatures_does_not_accept_partial_characters(self):
        """A fragment must not cut through the letters represented by one PDF glyph."""
        self.assertIsNone(SourceQuoteIndex('ﬀ').match('f'))
        self.assertEqual(SourceQuoteIndex('ﬀ').match('ff').text, 'ﬀ')


    def test_catalogue_checker_preserves_scientific_characters(self):
        """Apply the review checker to catalogue quotes without stripping scientific distinctions."""
        pairs = [
            ('X >= 1', 'X <= 1'), ('α ≤ 0.05', 'α < 0.05'),
            ('X = 1', 'x = 1'), ('β = 1', 'α = 1'),
            ('95% coverage', '75% coverage'),
            ('The method does not require overlap.', 'The method does require overlap.'),
            ('calibartion procedure', 'calibration procedure'),
            ('SUTVA is required.', 'sutva is required.'),
            ('X &gt;= 1', 'X <= 1'),
        ]
        for text, quote in pairs:
            with self.subTest(text=text, quote=quote):
                verified, valid = verify_evidence([{"source": "Abstract", "quote": quote}],
                    [EvidenceSection("Abstract", text)])
                self.assertFalse(valid)
                self.assertEqual(verified, [])

    def test_catalogue_typography_and_initial_case_retain_actual_spans(self):
        """Keep original offsets, source text and model quotes after harmless rendering repairs."""
        text = '  Preface. a “release” eﬀect\n for AT&amp;T.  '
        quote = 'A "release" effect for AT&T.'
        verified, valid = verify_evidence([{"source": "abstract", "quote": quote}],
            [EvidenceSection("Abstract", text)])
        self.assertTrue(valid)
        evidence = verified[0]
        self.assertEqual(evidence['model_quote'], quote)
        self.assertEqual(evidence['quote_grounding'], QUOTE_GROUNDING)
        self.assertEqual(evidence['quote'], 'a “release” eﬀect\n for AT&amp;T.')
        span = evidence['source_spans'][0]
        self.assertEqual(text[span['start']:span['end']], evidence['quote'])
        self.assertEqual(span['method'], 'initial_case')

    def test_catalogue_matches_only_text_visible_in_the_prompt(self):
        """Reject quotes from omitted sections or beyond the shared prompt budget."""
        sections = [EvidenceSection('README', '  First. Secret.  '), EvidenceSection('Documentation', 'Hidden.')]
        self.assertEqual(format_evidence(sections, 6), '--- README ---\nFirst.')
        for source, quote in [('README', 'Secret.'), ('Documentation', 'Hidden.')]:
            with self.subTest(source=source):
                self.assertFalse(verify_evidence([{'source': source, 'quote': quote}], sections, 6)[1])
        verified, valid = verify_evidence([{'source': 'README', 'quote': 'First.'}], sections, 6)
        self.assertTrue(valid)
        self.assertEqual(verified[0]['source_spans'][0]['start'], 2)

    def test_catalogue_ellipsis_preserves_order_and_cannot_reuse_a_span(self):
        """Ground repeated fragments independently and keep literal ellipses as a single span."""
        index = SourceQuoteIndex('The method. Other details. The method.')
        matches = index.match_fragments('The method. … The method.')
        self.assertEqual([match.start for match in matches], [0, 27])
        self.assertIsNone(SourceQuoteIndex('The method.').match_fragments('The method. … The method.'))
        self.assertIsNone(SourceQuoteIndex('X >= 1. Then Y.').match_fragments('X <= 1 … Then Y.'))
        self.assertEqual(len(SourceQuoteIndex('X … Y').match_fragments('X … Y')), 1)
        self.assertEqual(SourceQuoteIndex('The method.').match_fragments('…The method.…')[0].text, 'The method.')
        self.assertEqual(len(index.match_fragments('The method. &hellip; The method.')), 2)
        self.assertIsNone(index.match_fragments('…'))
        self.assertIsNone(SourceQuoteIndex('a method. a method.').match_fragments('A method … method.'))

    def test_html_entities_cannot_be_split_or_change_operators(self):
        """Map decoded entity characters to the entire original source entity."""
        index = SourceQuoteIndex('α &le; 1; &NotEqualTilde;')
        match = index.match('α ≤ 1')
        self.assertEqual(match.text, 'α &le; 1')
        self.assertIsNone(index.match('α < 1'))
        self.assertIsNone(index.match('≂'))
        self.assertEqual(index.match('≂̸').text, '&NotEqualTilde;')

    def test_ambiguous_source_labels_are_not_grounded(self):
        """Do not guess which source is intended when normalized section labels collide."""
        sections = [EvidenceSection('README', 'First.'), EvidenceSection('readme', 'Second.')]
        self.assertFalse(verify_evidence([{'source': 'readme', 'quote': 'Second.'}], sections)[1])

    def test_quote_policy_is_part_of_catalogue_cache_identity(self):
        """Prevent decisions made under another quote-validation policy from being reused."""
        current = decision_profile('model', 'prompt', {})
        with patch('radar.common.ollama.QUOTE_GROUNDING', 'another-policy'):
            self.assertNotEqual(current, decision_profile('model', 'prompt', {}))


    def test_pdf_line_wraps_preserve_hyphens_and_original_offsets(self):
        """Accept the observed decision-making and model-based wrapping without changing source text."""
        for text, quote in [
            ('Personalized decision-\nmaking is useful.', 'Personalized decision-making is useful.'),
            ('A conformal diffusion model-\n based approach.', 'A conformal diffusion model-based approach.'),
            ('The method is model-\r\n\tbased.', 'The method is model-based.'),
        ]:
            with self.subTest(text=text):
                match = SourceQuoteIndex(text).match(quote)
                self.assertIsNotNone(match)
                self.assertEqual(text[match.start:match.end], text)
                self.assertEqual(match.text, text)
                self.assertEqual(match.method, 'line_wrap')

    def test_discretionary_hyphens_require_a_corroborated_prose_word(self):
        """Only join split word parts when the complete word occurs elsewhere in the source."""
        text = 'The inference method. We use conformal in-\nference and supervised regression set-\nting. The setting is supervised.'
        match = SourceQuoteIndex(text).match('We use conformal inference and supervised regression setting.')
        self.assertIsNotNone(match)
        self.assertEqual(match.text, 'We use conformal in-\nference and supervised regression set-\nting.')
        self.assertIsNone(SourceQuoteIndex('A previously unseen made-\nupword.').match('A previously unseen madeupword.'))
        self.assertIsNone(SourceQuoteIndex('The inference method is described in- ference.').match('described inference.'))

    def test_line_wraps_never_remove_mathematical_operators_or_rewrite_words(self):
        """Keep variables, subtraction, numeric signs, math context and model edits fail-closed."""
        cases = [
            ('We measure X-\nY.', 'We measure XY.'),
            ('We measure x-\ny.', 'We measure xy.'),
            ('The result is 10-\n5.', 'The result is 105.'),
            ('The result is α-\nβ.', 'The result is αβ.'),
            ('The result is alpha-\nbeta.', 'The result is alphabeta.'),
            ('We call it alphabeta. alpha-\nbeta = 1.', 'alphabeta = 1.'),
            ('The result is X −\nY.', 'The result is XY.'),
            ('A model-\nbased approach requires X >= 1.', 'A model-based approach requires X <= 1.'),
            ('A model-\nbased calibartion.', 'A model-based calibration.'),
        ]
        for text, quote in cases:
            with self.subTest(quote=quote):
                self.assertIsNone(SourceQuoteIndex(text).match(quote))

    def test_line_wraps_and_initial_case_share_the_same_ambiguity_rule(self):
        """Preserve unique initial-capitalization matching across literal and wrapped passages."""
        text = 'We use a model-\nbased approach.'
        match = SourceQuoteIndex(text).match('A model-based approach.')
        self.assertEqual(match.text, 'a model-\nbased approach.')
        self.assertEqual(match.method, 'initial_case_line_wrap')
        repeated = 'a model-based approach. Again, a model-\nbased approach.'
        self.assertIsNone(SourceQuoteIndex(repeated).match('A model-based approach.'))

    def test_catalogue_line_wraps_keep_full_section_offsets(self):
        """The shared catalogue checker audits the same source-backed PDF wrapping repair."""
        text = '  Preface. A decision-\nmaking method.  '
        verified, valid = verify_evidence([{'source': 'Abstract', 'quote': 'A decision-making method.'}],
            [EvidenceSection('Abstract', text)])
        self.assertTrue(valid)
        evidence = verified[0]
        self.assertEqual(evidence['quote'], 'A decision-\nmaking method.')
        span = evidence['source_spans'][0]
        self.assertEqual(text[span['start']:span['end']], evidence['quote'])


    def test_unrelated_pdf_null_glyph_does_not_disable_line_wrap_matching(self):
        """Real PDF extracts can contain null glyphs outside a valid quoted passage."""
        text = 'An unreadable symbol: \x00. A decision-\nmaking method.'
        match = SourceQuoteIndex(text).match('A decision-making method.')
        self.assertEqual(match.text, 'A decision-\nmaking method.')


    def test_initial_case_uses_the_first_visible_character_after_rendering(self):
        """Preserve initial-article matching with encoded letters and leading soft hyphens."""
        for quote in ('&#65; model-based approach.', '\u00adA model-based approach.'):
            with self.subTest(quote=quote):
                self.assertEqual(SourceQuoteIndex('a model-based approach.').match(quote).text, 'a model-based approach.')
