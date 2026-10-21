"""Protect exact source ranges, cross-paragraph references and extraction identity."""
import json
import unittest
from unittest.mock import Mock, patch

from radar.common.evidence_blocks import EvidenceBlockIndex, MAX_BLOCK_CHARS, clip_structure
from radar.common.pdf import _tei_document, download_pdf_text
from radar.common.evidence import merge_new_evidence


class EvidenceBlockTests(unittest.TestCase):
    def test_blank_lines_separate_blocks_but_soft_wraps_and_typography_are_preserved(self):
        """Verify blank lines separate blocks but soft wraps and typography are preserved."""
        text = '  First para with calibartion and eﬀects.\nStill the same para.\n\nSecond α ≤ 0.05.\n\nThird treat-\nment.'
        index = EvidenceBlockIndex(text)
        self.assertEqual(len(index.blocks), 3)
        self.assertIn('\nStill', index.blocks[0].text)
        self.assertIn('treat-\nment', index.blocks[2].text)
        for block in index.blocks:
            self.assertEqual(text[block.start:block.end], block.text)
        self.assertEqual(index.fingerprint, EvidenceBlockIndex(text).fingerprint)
        self.assertNotEqual(index.blocks[0].id, EvidenceBlockIndex(text + ' Changed.').blocks[0].id)

    def test_multiblock_manifest_preserves_gaps_and_orders_passages_without_stitching(self):
        """Verify multiblock manifest preserves gaps and orders passages without stitching."""
        index = EvidenceBlockIndex('First claim.\n\nUnselected text.\n\nDistant supporting theorem.')
        ids = [index.blocks[2].id, index.blocks[0].id]
        self.assertIsNone(index.judgment_error({'value':'yes', 'evidence_ids':ids}, ['yes','unknown']))
        manifest = index.manifest([{'evidence_ids':ids}])
        self.assertEqual([p['text'] for p in manifest['passages']], ['First claim.', 'Distant supporting theorem.'])
        self.assertEqual(manifest['text_sha256'], index.text_sha256)
        self.assertNotIn('Unselected text.', json.dumps(manifest))

    def test_forged_foreign_missing_or_duplicate_ids_cannot_support_a_judgment(self):
        """Verify forged foreign missing or duplicate ids cannot support a judgment."""
        index = EvidenceBlockIndex('Original article.')
        other = EvidenceBlockIndex('Other article.')
        for ids in ([], ['invented'], [other.blocks[0].id], [index.blocks[0].id]*2):
            with self.subTest(ids=ids):
                self.assertIsNotNone(index.judgment_error({'value':'yes','evidence_ids':ids}, ['yes','unknown']))
        self.assertIsNone(index.judgment_error({'value':'unknown','evidence_ids':[]}, ['yes','unknown']))
        self.assertIsNotNone(index.judgment_error({'value':'unknown','evidence_ids':[index.blocks[0].id]}, ['yes','unknown']))
        self.assertIsNotNone(index.judgment_error({'value':'yes','quote':'Invented prose'}, ['yes','unknown']))
        with self.assertRaises(ValueError):
            index.manifest([{'evidence_ids':[other.blocks[0].id]}])

    def test_long_unstructured_pages_are_bounded_without_losing_nonwhitespace_characters(self):
        """Verify long unstructured pages are bounded without losing nonwhitespace characters."""
        for text in ('One long sentence. ' * 400, 'X' * (MAX_BLOCK_CHARS * 2 + 5)):
            index = EvidenceBlockIndex(text)
            self.assertTrue(all(len(block.text) <= MAX_BLOCK_CHARS for block in index.blocks))
            self.assertEqual(''.join(''.join(b.text.split()) for b in index.blocks), ''.join(text.split()))

    def test_structural_annotations_preserve_pages_sections_and_change_the_fingerprint(self):
        """Verify structural annotations preserve pages sections and change the fingerprint."""
        text = 'First page.\n\nSecond page.'
        structure = [{'start':0,'end':11,'kind':'page','page':1},
            {'start':13,'end':25,'kind':'page','page':2}]
        index = EvidenceBlockIndex(text, structure)
        self.assertEqual([b.page for b in index.blocks], [1,2])
        self.assertNotEqual(index.fingerprint, EvidenceBlockIndex(text).fingerprint)
        self.assertEqual(clip_structure(structure, 17)[1]['end'], 17)
        self.assertEqual(EvidenceBlockIndex(text[:17], clip_structure(structure,17)).blocks[-1].text, 'Seco')

    def test_structural_ranges_cannot_skip_or_reorder_source_content(self):
        """Verify structural ranges cannot skip or reorder source content."""
        for structure in ({}, [{'start':1,'end':4}], [{'start':0,'end':2}],
                [{'start':False,'end':4}], [{'start':0,'end':10}],
                [{'start':0,'end':3},{'start':1,'end':4}]):
            with self.subTest(structure=structure), self.assertRaises(ValueError):
                EvidenceBlockIndex('Text', structure)

    def test_tei_uses_real_paragraphs_headings_and_formula_boundaries(self):
        """Verify tei uses real paragraphs headings and formula boundaries."""
        data = b'<TEI><text><body><div><head>Methods</head><p>First paragraph.</p><p>Second paragraph.</p><formula>x = 1</formula></div></body></text></TEI>'
        text, structure = _tei_document(data)
        index = EvidenceBlockIndex(text, structure)
        self.assertEqual([b.kind for b in index.blocks], ['head','paragraph','paragraph','formula'])
        self.assertTrue(all(b.section == 'Methods' for b in index.blocks))
        self.assertTrue(all(text[b.start:b.end] == b.text for b in index.blocks))

    def test_pdf_retains_page_numbers_and_clips_metadata_with_the_text(self):
        """Verify pdf retains page numbers and clips metadata with the text."""
        document = Mock(pages=[Mock(extract_text=Mock(return_value='First page.')),
            Mock(extract_text=Mock(return_value='Second page.'))])
        with patch('radar.common.pdf._download_bytes',return_value=b'%PDF-test'), patch('pypdf.PdfReader',return_value=document):
            result = download_pdf_text('https://example.test/paper.pdf', 17)
        index = EvidenceBlockIndex.from_record(result)
        self.assertEqual([b.page for b in index.blocks], [1,2])
        self.assertEqual(index.blocks[-1].text, 'Seco')
        self.assertFalse(result['full_text_complete'])

    def test_expansion_does_not_reuse_structure_from_another_text(self):
        """Replacing text invalidates old page ranges even when the provider supplies no structure."""
        record = {'full_text': 'Old PDF page.', 'full_text_structure': [{'start':0,'end':13,'kind':'page','page':1}]}
        self.assertTrue(merge_new_evidence(record, {'full_text':'New text.'}))
        self.assertEqual(record['full_text_structure'], [])
        self.assertEqual(EvidenceBlockIndex.from_record(record).blocks[0].text, 'New text.')


if __name__ == '__main__':
    unittest.main()
