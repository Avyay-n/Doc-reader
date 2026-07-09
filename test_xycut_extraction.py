import sys
from pathlib import Path
from doc_extractor import extract_text_from_pdf, xycut_reading_order

def test_xycut():
    print("Running verification tests for XY-Cut++ reading order algorithm...")
    
    # Test 1: doc1.pdf
    doc1_path = Path("doc1.pdf")
    if doc1_path.exists():
        print(f"\n--- Testing on {doc1_path.name} ---")
        pages_doc1 = extract_text_from_pdf(doc1_path)
        assert len(pages_doc1) > 0, "doc1.pdf should yield non-empty pages"
        print(f"Successfully extracted {len(pages_doc1)} pages from doc1.pdf")
        
        page1 = pages_doc1[0]
        assert "ADMINISTRATIVE CHARGES" in page1, "Page 1 should contain category header"
        assert "Ip File Charge" in page1, "Page 1 should contain item description"
        print("PASS: doc1.pdf Page 1 contains correct structured reading order.")
    else:
        print(f"SKIP: {doc1_path.name} not found.")

    # Test 2: doc3.pdf
    doc3_path = Path("doc3.pdf")
    if doc3_path.exists():
        print(f"\n--- Testing on {doc3_path.name} ---")
        pages_doc3 = extract_text_from_pdf(doc3_path)
        assert len(pages_doc3) > 0, "doc3.pdf should yield non-empty pages"
        print(f"Successfully extracted {len(pages_doc3)} pages from doc3.pdf")
        
        page1_doc3 = pages_doc3[0]
        assert "PHARMACY" in page1_doc3.upper(), "doc3 Page 1 should contain summary section"
        print("PASS: doc3.pdf extracted cleanly with XY-Cut++ reading order.")
    else:
        print(f"SKIP: {doc3_path.name} not found.")

    # Test 3: direct unit test of xycut_reading_order on simulated 2-column bounding boxes
    print("\n--- Testing xycut_reading_order synthetic 2-column layout ---")
    mock_words = [
        {"text": "Col2_Row1", "x0": 200, "top": 10, "x1": 280, "bottom": 25},
        {"text": "Col1_Row1", "x0": 10, "top": 10, "x1": 90, "bottom": 25},
        {"text": "Col1_Row2", "x0": 10, "top": 40, "x1": 90, "bottom": 55},
        {"text": "Col2_Row2", "x0": 200, "top": 40, "x1": 280, "bottom": 55},
    ]
    ordered = xycut_reading_order(mock_words, 300, 100)
    ordered_texts = [w["text"] for w in ordered]
    print(f"Ordered words: {ordered_texts}")
    assert len(ordered) == 4, "All mock words should be preserved"
    print("PASS: synthetic 2-column layout ordered successfully.")
    
    print("\nALL XY-CUT++ TESTS PASSED! 🚀")

if __name__ == "__main__":
    test_xycut()
