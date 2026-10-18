"""
Interleaved Multimodal Document Formatter.
Serializes mixed text-image sequences into standard interleaved multimodal documents
with spatial coordinates, S3 URIs, and control tokens for Universal BPE tokenization.
"""

import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from schemas.data_schema import MediaReference


class InterleavedMultimodalFormatter:
    """
    Formats text documents with embedded visual anchors.
    Generates standard interleaved syntax:
      <|image_ref_start|>id="..." uri="..." width="..." height="..." align="..." alt="..."<|image_ref_end|>
    Preserves document reading order and cross-referencing captions.
    """

    def __init__(
        self,
        token_prefix: str = "<|image_ref_start|>",
        token_suffix: str = "<|image_ref_end|>"
    ):
        self.token_prefix = token_prefix
        self.token_suffix = token_suffix

    def format_media_tag(
        self,
        image_id: str,
        s3_uri: str,
        alt_text: str,
        width: int = 1024,
        height: int = 768,
        mime_type: str = "image/jpeg",
        alignment_score: float = 0.32,
        bounding_box: Optional[List[int]] = None
    ) -> str:
        """
        Constructs the structured multimodal media anchor tag.
        """
        bbox_str = f" bbox=\"{','.join(map(str, bounding_box))}\"" if bounding_box else ""
        escaped_alt = alt_text.replace('"', '&quot;')

        tag = (
            f"{self.token_prefix} "
            f"id=\"{image_id}\" "
            f"uri=\"{s3_uri}\" "
            f"mime=\"{mime_type}\" "
            f"width=\"{width}\" "
            f"height=\"{height}\" "
            f"align=\"{alignment_score:.3f}\"{bbox_str} "
            f"alt=\"{escaped_alt}\" "
            f"{self.token_suffix}"
        )
        return tag

    def format_interleaved_document(
        self,
        document_id: str,
        text_paragraphs: List[str],
        media_items: List[Dict[str, Any]],
        s3_bucket_prefix: str = "s3://ai-data-shards/multimodal/images"
    ) -> Dict[str, Any]:
        """
        Interleaves text paragraphs with validated visual media tags in document order.
        """
        interleaved_parts = []
        image_manifest = []

        # Position media items across paragraphs
        num_paras = max(1, len(text_paragraphs))
        media_count = len(media_items)

        # Distribute images evenly between paragraphs
        placed_media = 0
        for p_idx, para in enumerate(text_paragraphs):
            interleaved_parts.append(para)

            # Check if an image should be placed after this paragraph
            if placed_media < media_count and (p_idx + 1) * media_count // num_paras > placed_media:
                media = media_items[placed_media]
                img_hash = hashlib.sha256(f"{document_id}_{placed_media}_{media.get('url', '')}".encode()).hexdigest()[:12]
                img_id = f"img_{document_id[:8]}_{placed_media:02d}_{img_hash}"
                s3_uri = f"{s3_bucket_prefix}/{img_id}.jpg"

                alt = media.get("alt_text", "")
                w = media.get("width", 1024)
                h = media.get("height", 768)
                score = media.get("alignment_score", 0.30)
                mime = media.get("mime_type", "image/jpeg")
                bbox = media.get("bounding_box", None)

                tag = self.format_media_tag(
                    image_id=img_id,
                    s3_uri=s3_uri,
                    alt_text=alt,
                    width=w,
                    height=h,
                    mime_type=mime,
                    alignment_score=score,
                    bounding_box=bbox
                )
                interleaved_parts.append(tag)

                image_manifest.append({
                    "image_id": img_id,
                    "s3_uri": s3_uri,
                    "source_url": media.get("url", ""),
                    "alt_text": alt,
                    "width": w,
                    "height": h,
                    "alignment_score": score,
                    "mime_type": mime,
                    "paragraph_index": p_idx
                })
                placed_media += 1

        # Any remaining media items appended at the end
        while placed_media < media_count:
            media = media_items[placed_media]
            img_hash = hashlib.sha256(f"{document_id}_{placed_media}".encode()).hexdigest()[:12]
            img_id = f"img_{document_id[:8]}_{placed_media:02d}_{img_hash}"
            s3_uri = f"{s3_bucket_prefix}/{img_id}.jpg"
            tag = self.format_media_tag(
                image_id=img_id,
                s3_uri=s3_uri,
                alt_text=media.get("alt_text", ""),
                width=media.get("width", 1024),
                height=media.get("height", 768),
                alignment_score=media.get("alignment_score", 0.30)
            )
            interleaved_parts.append(tag)
            image_manifest.append({
                "image_id": img_id,
                "s3_uri": s3_uri,
                "source_url": media.get("url", ""),
                "alt_text": media.get("alt_text", "")
            })
            placed_media += 1

        full_interleaved_text = "\n\n".join(interleaved_parts)

        return {
            "document_id": document_id,
            "full_interleaved_text": full_interleaved_text,
            "total_images": len(image_manifest),
            "image_manifest": image_manifest,
            "text_paragraph_count": len(text_paragraphs)
        }
