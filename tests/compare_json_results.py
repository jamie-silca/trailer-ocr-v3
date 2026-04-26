import json
import argparse
import os
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np

def calculate_iou(box1, box2):
    """
    box: [x, y, w, h] (COCO format)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def normalize_text(text):
    if not text:
        return ""
    return text.strip().upper()

def compare_results(gt_json_path, auto_json_path, iou_threshold=0.5):
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(auto_json_path, 'r', encoding='utf-8') as f:
        auto_data = json.load(f)

    # Map image IDs to file names for reporting
    img_id_to_name = {img['id']: img['file_name'] for img in gt_data['images']}
    
    # Organize annotations by image_id
    gt_annots = defaultdict(list)
    for ann in gt_data['annotations']:
        gt_annots[ann['image_id']].append(ann)
        
    auto_annots = defaultdict(list)
    for ann in auto_data['annotations']:
        auto_annots[ann['image_id']].append(ann)

    all_image_ids = set(gt_annots.keys()) | set(auto_annots.keys())
    
    results = {
        'total_gt': 0,
        'total_auto': 0,
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'matched_pairs': [], # list of (gt_ann, auto_ann, iou)
        'missed_gt': [],
        'hallucinated_auto': []
    }

    for img_id in sorted(all_image_ids):
        gts = gt_annots.get(img_id, [])
        autos = auto_annots.get(img_id, [])
        
        results['total_gt'] += len(gts)
        results['total_auto'] += len(autos)
        
        matched_gt_indices = set()
        matched_auto_indices = set()
        
        # Greedy matching based on IoU
        # Find all possible overlaps
        overlaps = []
        for i, gt in enumerate(gts):
            for j, auto in enumerate(autos):
                iou = calculate_iou(gt['bbox'], auto['bbox'])
                if iou >= iou_threshold:
                    overlaps.append((iou, i, j))
        
        # Sort by IoU descending
        overlaps.sort(key=lambda x: x[0], reverse=True)
        
        for iou, i, j in overlaps:
            if i not in matched_gt_indices and j not in matched_auto_indices:
                matched_gt_indices.add(i)
                matched_auto_indices.add(j)
                results['tp'] += 1
                results['matched_pairs'].append({
                    'image_id': img_id,
                    'file_name': img_id_to_name.get(img_id, "unknown"),
                    'gt': gts[i],
                    'auto': autos[j],
                    'iou': iou
                })
        
        # Unmatched GTs are False Negatives
        for i, gt in enumerate(gts):
            if i not in matched_gt_indices:
                results['fn'] += 1
                results['missed_gt'].append({
                    'image_id': img_id,
                    'file_name': img_id_to_name.get(img_id, "unknown"),
                    'gt': gt
                })
                
        # Unmatched Autos are False Positives
        for j, auto in enumerate(autos):
            if j not in matched_auto_indices:
                results['fp'] += 1
                results['hallucinated_auto'].append({
                    'image_id': img_id,
                    'file_name': img_id_to_name.get(img_id, "unknown"),
                    'auto': auto
                })

    # OCR Evaluation on TPs
    total_cer = 0
    exact_matches = 0
    
    for pair in results['matched_pairs']:
        gt_text = normalize_text(pair['gt'].get('text', ''))
        auto_text = normalize_text(pair['auto'].get('text', ''))
        
        dist = levenshtein_distance(gt_text, auto_text)
        cer = dist / max(len(gt_text), 1)
        
        pair['cer'] = cer
        pair['levenshtein'] = dist
        pair['gt_text_norm'] = gt_text
        pair['auto_text_norm'] = auto_text
        
        total_cer += cer
        if gt_text == auto_text:
            exact_matches += 1
            pair['exact_match'] = True
        else:
            pair['exact_match'] = False

    # Summarize
    summary = {
        'precision': results['tp'] / (results['tp'] + results['fp']) if (results['tp'] + results['fp']) > 0 else 0,
        'recall': results['tp'] / (results['tp'] + results['fn']) if (results['tp'] + results['fn']) > 0 else 0,
        'tp': results['tp'],
        'fp': results['fp'],
        'fn': results['fn'],
        'exact_match_rate': exact_matches / results['tp'] if results['tp'] > 0 else 0,
        'avg_cer': total_cer / results['tp'] if results['tp'] > 0 else 0,
        'avg_iou': np.mean([p['iou'] for p in results['matched_pairs']]) if results['matched_pairs'] else 0
    }
    summary['f1'] = 2 * (summary['precision'] * summary['recall']) / (summary['precision'] + summary['recall']) if (summary['precision'] + summary['recall']) > 0 else 0
    
    return results, summary

def generate_report(results, summary, gt_path, auto_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Empirical OCR Comparison Report\n\n")
        f.write(f"- **Ground Truth**: `{os.path.basename(gt_path)}`\n")
        f.write(f"- **Auto Results**: `{os.path.basename(auto_path)}`\n\n")
        
        f.write("## Summary Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| Precision | {summary['precision']:.4f} |\n")
        f.write(f"| Recall | {summary['recall']:.4f} |\n")
        f.write(f"| F1-Score | {summary['f1']:.4f} |\n")
        f.write(f"| Avg IoU | {summary['avg_iou']:.4f} |\n")
        f.write(f"| Exact Match Rate (EMR) | {summary['exact_match_rate']:.2%} |\n")
        f.write(f"| Avg CER | {summary['avg_cer']:.4f} |\n")
        f.write(f"| True Positives (Matches) | {summary['tp']} |\n")
        f.write(f"| False Positives (Hallucinations) | {summary['fp']} |\n")
        f.write(f"| False Negatives (Missed) | {summary['fn']} |\n\n")
        
        f.write("## OCR Discrepancies (Top 20 by CER)\n\n")
        discrepancies = [p for p in results['matched_pairs'] if not p['exact_match']]
        discrepancies.sort(key=lambda x: x['cer'], reverse=True)
        
        f.write("| Image | GT Text | Auto Text | CER | IoU |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for p in discrepancies[:20]:
            f.write(f"| {p['file_name']} | `{p['gt_text_norm']}` | `{p['auto_text_norm']}` | {p['cer']:.4f} | {p['iou']:.2f} |\n")
            
        f.write("\n## Missed Detections (False Negatives - First 20)\n\n")
        for fn in results['missed_gt'][:20]:
            text = fn['gt'].get('text', 'N/A')
            f.write(f"- **{fn['file_name']}**: Missed `{text}` at box {fn['gt']['bbox']}\n")
            
        f.write("\n## Hallucinations (False Positives - First 20)\n\n")
        for fp in results['hallucinated_auto'][:20]:
            text = fp['auto'].get('text', 'N/A')
            f.write(f"- **{fp['file_name']}**: Predicted `{text}` at box {fp['auto']['bbox']}\n")

    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--auto", required=True)
    parser.add_argument("--output", default="tests/results/comparison_report.md")
    args = parser.parse_args()
    
    results, summary = compare_results(args.gt, args.auto)
    generate_report(results, summary, args.gt, args.auto, args.output)
