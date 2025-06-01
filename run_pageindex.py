"""Command line interface for running PageIndex."""

import argparse
import json
import os

from pageindex import page_index
from pageindex.utils import ConfigLoader

def parse_args() -> argparse.Namespace:
    """Return command line arguments."""

    parser = argparse.ArgumentParser(
        description="Process a PDF document and generate its structure",
    )
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20", help="Model to use")
    parser.add_argument(
        "--toc-check-pages",
        type=int,
        default=20,
        help="Number of pages to check for table of contents",
    )
    parser.add_argument(
        "--max-pages-per-node",
        type=int,
        default=10,
        help="Maximum number of pages per node",
    )
    parser.add_argument(
        "--max-tokens-per-node",
        type=int,
        default=20000,
        help="Maximum number of tokens per node",
    )
    parser.add_argument(
        "--if-add-node-id",
        choices=["yes", "no"],
        default="yes",
        help="Whether to add node id to each node",
    )
    parser.add_argument(
        "--if-add-node-summary",
        choices=["yes", "no"],
        default="no",
        help="Whether to add a summary to each node",
    )
    parser.add_argument(
        "--if-add-doc-description",
        choices=["yes", "no"],
        default="yes",
        help="Whether to add a document description",
    )
    parser.add_argument(
        "--if-add-node-text",
        choices=["yes", "no"],
        default="no",
        help="Whether to include raw text in nodes",
    )
    return parser.parse_args()


def main() -> None:
    """Run PageIndex using CLI arguments."""

    args = parse_args()

    opt = ConfigLoader().load(
        {
            "model": args.model,
            "toc_check_page_num": args.toc_check_pages,
            "max_page_num_each_node": args.max_pages_per_node,
            "max_token_num_each_node": args.max_tokens_per_node,
            "if_add_node_id": args.if_add_node_id,
            "if_add_node_summary": args.if_add_node_summary,
            "if_add_doc_description": args.if_add_doc_description,
            "if_add_node_text": args.if_add_node_text,
        }
    )

    toc_with_page_number = page_index(args.pdf_path, **vars(opt))

    print("Parsing done, saving to file...")
    pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
    os.makedirs("./results", exist_ok=True)
    with open(f"./results/{pdf_name}_structure.json", "w", encoding="utf-8") as f:
        json.dump(toc_with_page_number, f, indent=2)


if __name__ == "__main__":
    main()
