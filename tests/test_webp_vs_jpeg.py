"""
Compare detect-and-ocr results between original JPEG and medium WebP (1600x1600).
Tests Layer 1 of OOM fix: send medium WebP instead of full-res to save memory.

Usage:
    python tests/test_webp_vs_jpeg.py
"""
import json, os, io, time, sys
import urllib.request, urllib.parse

try:
    from PIL import Image
except ImportError:
    sys.exit("PIL not available — run from inside the trailer-ocr-v3 venv or install Pillow")

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "20260420")
GT_FILE = os.path.join(DATASET_DIR, "annotations_2026-04-20_12-50_coco_with_text.json")
SERVICE_URL = os.environ.get("SERVICE_URL", "http://localhost:8001")
ENDPOINT = f"{SERVICE_URL}/detect-and-ocr"

MEDIUM_SIZE = 1600  # match production medium WebP dims
TEST_IMAGES = [
    "DJI_20260420125037_0001_V.jpeg",  # 2 detections: 772, R50453
    "DJI_20260420125039_0002_V.jpeg",  # 2 detections: 473247, 73247
    "DJI_20260420125041_0003_V.jpeg",  # 3 detections: 473120, 473135, RR2177
    "DJI_20260420125043_0004_V.jpeg",  # 3 detections: 473120, 473135, R76449
    "DJI_20260420125049_0007_V.jpeg",  # 3 detections: 7336, R41600, RR2442
]


def make_medium_webp(jpeg_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(jpeg_bytes))
    img = img.convert("RGB")
    img.thumbnail((MEDIUM_SIZE, MEDIUM_SIZE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=80)
    return buf.getvalue()


def post_image(image_bytes: bytes, filename: str) -> dict:
    boundary = "----FormBoundary"
    body_parts = []
    body_parts.append(f"--{boundary}\r\n".encode())
    body_parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    content_type = "image/webp" if filename.endswith(".webp") else "image/jpeg"
    body_parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
    body_parts.append(image_bytes)
    body_parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(body_parts)

    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()}
    except Exception as e:
        return {"error": str(e)}


def load_ground_truth():
    with open(GT_FILE) as f:
        d = json.load(f)
    img_name_to_id = {i["file_name"]: i["id"] for i in d["images"]}
    annos_by_img = {}
    for a in d["annotations"]:
        iid = a["image_id"]
        annos_by_img.setdefault(iid, []).append(a)
    return img_name_to_id, annos_by_img


def fmt_size(n: int) -> str:
    return f"{n/1024:.1f} KB" if n < 1_000_000 else f"{n/1_048_576:.2f} MB"


def main():
    img_name_to_id, annos_by_gt = load_ground_truth()

    print(f"Service: {ENDPOINT}")
    print(f"Testing {len(TEST_IMAGES)} images\n")
    print(f"{'Image':<40} {'Format':<8} {'Size':>9}  {'#Det':>4}  {'Texts found':<40}  {'GT texts':<40}  {'Time':>7}")
    print("-" * 160)

    summary = {"jpeg_det": 0, "webp_det": 0, "gt_det": 0, "jpeg_text_hits": 0, "webp_text_hits": 0}

    for fname in TEST_IMAGES:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue

        with open(path, "rb") as f:
            jpeg_bytes = f.read()
        webp_bytes = make_medium_webp(jpeg_bytes)

        # Ground truth
        img_id = img_name_to_id.get(fname)
        gt_annos = annos_by_gt.get(img_id, [])
        gt_texts = sorted(set(a.get("text", "") for a in gt_annos if a.get("text")))
        summary["gt_det"] += len(gt_annos)

        for label, data, filename_send in [
            ("JPEG", jpeg_bytes, fname),
            ("WebP", webp_bytes, fname.replace(".jpeg", ".medium.webp")),
        ]:
            t0 = time.time()
            result = post_image(data, filename_send)
            elapsed = time.time() - t0

            if "error" in result:
                print(f"  {fname:<40} {label:<8} {fmt_size(len(data)):>9}  ERR   {result}")
                continue

            dets = result.get("detections", [])
            texts_found = sorted(set(d.get("text", "") for d in dets if d.get("text")))

            hits = len(set(texts_found) & set(gt_texts))
            print(
                f"  {fname:<40} {label:<8} {fmt_size(len(data)):>9}  {len(dets):>4}  "
                f"{str(texts_found):<40}  {str(gt_texts):<40}  {elapsed:>6.1f}s"
            )

            if label == "JPEG":
                summary["jpeg_det"] += len(dets)
                summary["jpeg_text_hits"] += hits
            else:
                summary["webp_det"] += len(dets)
                summary["webp_text_hits"] += hits

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  Ground truth detections : {summary['gt_det']}")
    print(f"  JPEG detections         : {summary['jpeg_det']}")
    print(f"  WebP detections         : {summary['webp_det']}")
    print(f"  JPEG text matches vs GT : {summary['jpeg_text_hits']}")
    print(f"  WebP text matches vs GT : {summary['webp_text_hits']}")
    webp_det_ratio = (summary["webp_det"] / summary["jpeg_det"] * 100) if summary["jpeg_det"] else 0
    print(f"  WebP/JPEG detection ratio: {webp_det_ratio:.1f}%  (>= 80% = acceptable)")
    if webp_det_ratio >= 80:
        print("  PASS - WebP medium is suitable for production detection")
    else:
        print("  FAIL - WebP medium loses too many detections vs full JPEG")


if __name__ == "__main__":
    main()
