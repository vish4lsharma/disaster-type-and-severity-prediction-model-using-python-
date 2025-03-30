import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DisasterAnalyzer:
    """Class for analyzing natural disasters in images with improved functionality"""

    def __init__(self):
        self.disaster_types = ['flood', 'fire', 'hurricane', 'landslide', 'snowstorm']
        self.disaster_colors = {
            'flood': [0, 0, 255],      # Blue
            'fire': [255, 0, 0],       # Red
            'hurricane': [128, 0, 128], # Purple
            'landslide': [139, 69, 19], # Brown
            'snowstorm': [255, 255, 255] # White
        }
        self.kernel = np.ones((5,5), np.uint8)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3])  # Ensure RGB channels only
        ])

    def _load_image(self, image_url: str) -> Optional[np.ndarray]:
        """Load and preprocess image from URL with improved error handling"""
        try:
            headers = {'User-Agent': 'DisasterAnalyzer/1.0'}
            response = requests.get(image_url, headers=headers, stream=True, timeout=10)
            response.raise_for_status()

            if 'image' not in response.headers.get('Content-Type', ''):
                logger.error(f"URL does not point to an image: {response.headers.get('Content-Type')}")
                return None

            img = Image.open(BytesIO(response.content)).convert('RGB')
            return np.array(img)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None

    def _convert_to_hsv(self, img_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert image to HSV with optimization"""
        img_pil = Image.fromarray(img_array)
        img_hsv = transforms.functional.to_tensor(img_pil.convert('HSV'))
        return {
            'h': img_hsv[0].numpy(),
            's': img_hsv[1].numpy(),
            'v': img_hsv[2].numpy()
        }

    def _analyze_disaster(self, disaster: str, hsv: Dict[str, np.ndarray],
                         img_array: np.ndarray, img_gray: np.ndarray) -> Dict:
        """Analyze specific disaster type with improved detection algorithms"""
        h, s, v = hsv['h'], hsv['s'], hsv['v']

        if disaster == 'flood':
            mask = (h > 0.5) & (h < 0.7) & (s > 0.2) & (s < 0.8) & (v < 0.7)
            severity_thresholds = [1, 5, 15]

        elif disaster == 'fire':
            red_orange = ((h < 0.05) | (h > 0.9)) & (s > 0.5) & (v > 0.5)
            yellow = (h > 0.05) & (h < 0.15) & (s > 0.5) & (v > 0.7)
            mask = red_orange | yellow
            intensity = np.mean(v[mask]) * 10 if np.sum(mask) > 0 else 0
            severity_thresholds = [1, 5, 15]

        elif disaster == 'hurricane':
            edges = cv2.Canny(cv2.GaussianBlur(img_gray, (5, 5), 0), 50, 150)
            circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=20, maxRadius=0)
            cloud_mask = (h > 0.5) & (h < 0.75) & (s < 0.3) & (v > 0.7)
            mask = cloud_mask & (cv2.dilate(edges, self.kernel, iterations=1) > 0)
            severity_thresholds = [5, 15, 35]

        elif disaster == 'landslide':
            brown = (h > 0.05) & (h < 0.15) & (s > 0.3) & (s < 0.8) & (v > 0.2) & (v < 0.8)
            green = (h > 0.2) & (h < 0.4) & (s > 0.3) & (v > 0.2)
            texture = cv2.Laplacian(img_gray, cv2.CV_64F)
            mask = brown & (np.abs(texture) > 20) & ~green
            severity_thresholds = [3, 10, 20]

        elif disaster == 'snowstorm':
            snow = (v > 0.7) & (s < 0.3)
            texture = cv2.Laplacian(cv2.GaussianBlur(img_gray, (5, 5), 0), cv2.CV_64F)
            mask = snow & (np.abs(texture) < 10) & (img_gray > 150)
            severity_thresholds = [10, 25, 50]

        # Clean mask
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel).astype(bool)

        # Calculate statistics
        percentage = (np.sum(mask) / mask.size) * 100
        severity_level = sum(percentage > t for t in severity_thresholds)
        severity = ["NONE", "MILD", "MODERATE", "SEVERE"][severity_level]

        result = {
            'percentage': percentage,
            'severity': severity,
            'severity_level': severity_level,
            'mask': mask
        }
        if disaster == 'fire':
            result['intensity'] = intensity

        return result

    def analyze_image(self, image_url: str, disaster_types: Optional[List[str]] = None) -> Dict:
        """Main analysis function with parallel processing"""
        start_time = time.time()

        if disaster_types is None:
            disaster_types = self.disaster_types
        else:
            disaster_types = [d.lower() for d in disaster_types]
            invalid_types = set(disaster_types) - set(self.disaster_types)
            if invalid_types:
                logger.warning(f"Invalid disaster types ignored: {invalid_types}")
                disaster_types = [d for d in disaster_types if d in self.disaster_types]

        # Load image
        img_array = self._load_image(image_url)
        if img_array is None:
            return {}

        # Preprocess image
        hsv = self._convert_to_hsv(img_array)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Parallel analysis
        with ThreadPoolExecutor() as executor:
            analysis_func = partial(self._analyze_disaster, hsv=hsv,
                                  img_array=img_array, img_gray=img_gray)
            results = dict(executor.map(lambda d: (d, analysis_func(d)), disaster_types))

        # Visualization
        self._visualize_results(img_array, results, disaster_types)

        # Log execution time
        logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        return results

    def _visualize_results(self, img_array: np.ndarray, results: Dict, disaster_types: List[str]):
        """Improved visualization with better layout and annotations"""
        num_disasters = len(disaster_types)
        fig = plt.figure(figsize=(15, 5 * (num_disasters + 1)))

        # Original image
        plt.subplot(num_disasters + 2, 2, 1)
        plt.imshow(img_array)
        plt.title("Original Image")
        plt.axis('off')

        # Combined overlay
        combined_overlay = np.zeros_like(img_array)
        combined_mask = np.zeros(img_array.shape[:2], dtype=bool)

        for disaster in sorted(disaster_types, key=lambda x: self.disaster_types.index(x)):
            if disaster in results:
                mask = results[disaster]['mask']
                apply_mask = mask & ~combined_mask
                combined_overlay[apply_mask] = self.disaster_colors[disaster]
                combined_mask |= mask

        alpha = 0.5
        combined_blended = np.clip((1-alpha) * img_array + alpha * combined_overlay, 0, 255).astype(np.uint8)

        plt.subplot(num_disasters + 2, 2, 2)
        plt.imshow(combined_blended)
        plt.title("Combined Detection")
        plt.axis('off')

        # Individual disasters
        for i, disaster in enumerate(disaster_types, 3):
            if disaster in results:
                overlay = np.zeros_like(img_array)
                overlay[results[disaster]['mask']] = self.disaster_colors[disaster]
                blended = np.clip((1-alpha) * img_array + alpha * overlay, 0, 255).astype(np.uint8)

                plt.subplot(num_disasters + 2, 2, i)
                plt.imshow(blended)
                plt.title(f"{disaster.capitalize()} Detection\n{results[disaster]['severity']}")
                plt.axis('off')

        # Legend
        ax = plt.subplot(num_disasters + 2, 2, num_disasters + 3)
        for i, disaster in enumerate(disaster_types):
            if disaster in results:
                color = np.array(self.disaster_colors[disaster])/255.0
                ax.add_patch(plt.Rectangle((0, i), 1, 0.8, color=color))
                text = f"{disaster.capitalize()}: {results[disaster]['severity']}"
                if disaster == 'fire':
                    text += f" (Intensity: {results[disaster]['intensity']:.1f})"
                ax.text(1.5, i+0.4, text, va='center')

        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, len(disaster_types) - 0.5)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        # Print results
        print("\n=== DISASTER ANALYSIS RESULTS ===")
        for disaster in disaster_types:
            if disaster in results:
                r = results[disaster]
                print(f"\n{disaster.upper()}:")
                print(f"- Coverage: {r['percentage']:.2f}%")
                if 'intensity' in r:
                    print(f"- Intensity: {r['intensity']:.1f}/10")
                print(f"- Severity: {r['severity']}")

def main():
    analyzer = DisasterAnalyzer()
    example_url = "https://images.nationalgeographic.org/image/upload/v1638889445/EducationHub/photos/flooding-in-bangladesh.jpg"

    # Full analysis
    print("Performing full analysis...")
    results = analyzer.analyze_image(example_url)

    # Specific analysis
    print("\nPerforming flood-specific analysis...")
    results = analyzer.analyze_image(example_url, ['flood', 'fire'])

if __name__ == "__main__":
    main()
