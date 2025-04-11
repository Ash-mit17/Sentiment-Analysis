import sys
import pkg_resources

required_packages = {
    'torch': '1.9.0',
    'transformers': '4.15.0',
    'timm': '0.5.4',
    'Pillow': '8.3.1',
    'imageio': '2.9.0',
    'imageio-ffmpeg': '0.4.5',
    'torchvision': '0.10.0',
    'pandas': '1.3.0',
    'numpy': '1.19.5',
    'scikit-learn': '0.24.2',
    'matplotlib': '3.4.3',
    'seaborn': '0.11.2',
    'tqdm': '4.62.0',
    'python-dotenv': '0.19.0',
    'tensorboard': '2.7.0'
}

def check_package(package_name, min_version):
    try:
        version = pkg_resources.get_distribution(package_name).version
        if pkg_resources.parse_version(version) >= pkg_resources.parse_version(min_version):
            print(f"✓ {package_name} {version} (>= {min_version})")
            return True
        else:
            print(f"✗ {package_name} {version} (needs >= {min_version})")
            return False
    except pkg_resources.DistributionNotFound:
        print(f"✗ {package_name} not installed")
        return False

def main():
    print("Checking required packages...\n")
    
    all_installed = True
    for package, version in required_packages.items():
        if not check_package(package, version):
            all_installed = False
    
    if all_installed:
        print("\nAll packages are installed correctly!")
    else:
        print("\nSome packages are missing or have incorrect versions.")
        print("Please run: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 