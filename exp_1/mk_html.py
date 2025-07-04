import os
import argparse
import re
import base64

def parse_predictions_file(text_file):
    """Parse the predictions.txt file into structured data"""
    sections = []
    overall_metrics = {}
    
    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # First split by the detailed predictions divider
    main_parts = re.split(r'DETAILED PREDICTIONS:\n={80}\n', content)
    
    # Parse overall metrics if exists
    if len(main_parts) > 0 and main_parts[0].strip().startswith('OVERALL EVALUATION METRICS'):
        metrics_part = main_parts[0].split('\n')[2:]  # Skip header and divider
        for line in metrics_part:
            if ':' in line:
                metric, value = line.split(':', 1)
                overall_metrics[metric.strip()] = value.strip()
    
    # Parse image sections if detailed predictions exist
    if len(main_parts) > 1:
        # Split individual image sections (including the first one)
        image_sections = re.split(r'\n={80}\n', main_parts[1])
        
        for section in image_sections:
            section = section.strip()
            if not section.startswith('Image:'):
                continue
                
            lines = [line.strip() for line in section.split('\n') if line.strip()]
            image_info = {
                'filename': re.search(r'Image:\s+([^\s]+\.jpg)', lines[0]).group(1),
                'predictions': [],
                'references': [],
                'metrics': {}
            }
            
            current_line = 1
            
            # Parse predictions
            if current_line < len(lines) and lines[current_line].startswith('Predictions:'):
                current_line += 1
                while (current_line < len(lines) and 
                       not lines[current_line].startswith('References:')):
                    image_info['predictions'].append(lines[current_line])
                    current_line += 1
            
            # Parse references
            if current_line < len(lines) and lines[current_line].startswith('References:'):
                current_line += 1
                while (current_line < len(lines) and 
                       not lines[current_line].startswith('Metrics for this image:')):
                    image_info['references'].append(lines[current_line])
                    current_line += 1
            
            # Parse metrics
            if current_line < len(lines) and lines[current_line].startswith('Metrics for this image:'):
                current_line += 1
                while current_line < len(lines) and ':' in lines[current_line]:
                    # Handle both ": " and ":" cases
                    if ': ' in lines[current_line]:
                        metric, value = lines[current_line].split(': ', 1)
                    else:
                        metric, value = lines[current_line].split(':', 1)
                    image_info['metrics'][metric.strip()] = value.strip()
                    current_line += 1
            
            sections.append(image_info)
    
    return overall_metrics, sections

def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Warning: Could not encode image {image_path} - {str(e)}")
        return None

def create_html_report(overall_metrics, sections, image_dir, output_file):
    """Create HTML file with images and predictions"""
    with open(output_file, 'w', encoding='utf-8') as html:
        # Write HTML header
        html.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Image Captioning Predictions Report</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.5; margin: 20px; }
        h1 { color: #333; }
        h2 { border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        .image-section { margin-bottom: 30px; }
        img { max-width: 400px; max-height: 400px; }
        .metrics { margin-top: 10px; }
        .metric { margin-left: 20px; }
        hr { border: 0; height: 1px; background: #ddd; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Image Captioning Predictions Report</h1>
""")

        # Write overall metrics if they exist
        if overall_metrics:
            html.write("    <h2>Overall Evaluation Metrics</h2>\n")
            for metric, value in overall_metrics.items():
                html.write(f"    <div>{metric}: {value}</div>\n")
            html.write("    <hr>\n")

        # Write each image section
        html.write("    <h2>Detailed Predictions</h2>\n")
        for i, section in enumerate(sections, 1):
            image_path = os.path.join(image_dir, section['filename'])
            image_data = image_to_base64(image_path)
            
            html.write(f"    <div class='image-section'>\n")
            html.write(f"        <h3>Image {i}: {section['filename']}</h3>\n")
            
            if image_data:
                html.write(f"        <img src='data:image/jpeg;base64,{image_data}' alt='{section['filename']}'><br>\n")
            else:
                html.write(f"        <p>Image not found: {section['filename']}</p>\n")
            
            html.write("        <h4>Predictions:</h4>\n")
            for pred in section['predictions']:
                html.write(f"        <div>{pred}</div>\n")
            
            html.write("        <h4>References:</h4>\n")
            for ref in section['references']:
                html.write(f"        <div>{ref}</div>\n")
            
            html.write("        <h4>Metrics:</h4>\n")
            html.write("        <div class='metrics'>\n")
            for metric, value in section['metrics'].items():
                html.write(f"            <div class='metric'>{metric}: {value}</div>\n")
            html.write("        </div>\n")
            
            html.write("    </div>\n")
            if i < len(sections):
                html.write("    <hr>\n")

        # Write HTML footer
        html.write("""</body>
</html>
""")
        
        print(f"Successfully created HTML report: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert predictions.txt to an HTML report with images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--text_file', required=True, 
                       help='Path to predictions.txt file')
    parser.add_argument('--image_dir', required=True, 
                       help='Directory containing the images')
    parser.add_argument('--output_file', required=True, 
                       help='Output HTML file path')
    
    args = parser.parse_args()
    
    print(f"Parsing predictions file: {args.text_file}")
    overall_metrics, sections = parse_predictions_file(args.text_file)
    print(f"Found {len(sections)} image sections in the file")
    
    print(f"Creating HTML report: {args.output_file}")
    create_html_report(overall_metrics, sections, args.image_dir, args.output_file)

if __name__ == "__main__":
    main()

