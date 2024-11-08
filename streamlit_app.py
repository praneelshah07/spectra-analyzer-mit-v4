import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.style import use
use('fast')
import json
import random
import zipfile
import io
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
import uuid

# ---------------------------
# Configuration and Styling
# ---------------------------

# Set page layout to 'wide' for full screen usage
st.set_page_config(page_title="Spectra Visualization App", layout="wide")

# Adding custom CSS to style the banner, headers, and descriptions
st.markdown("""
    <style>
    .banner {
        width: 100%;
        background-color: #89CFF0;  /* Sky Blue */
        color: white;
        padding: 20px;
        text-align: center;
        font-size: 45px;  
        font-weight: bold;
        margin-bottom: 20px;
    }
    .header {
        padding: 10px;
        margin-bottom: 20px;
        background-color: #f0f8ff;
        border: 2px solid #89CFF0;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .sidebar-title {
        font-size: 25px;  
        font-weight: bold;
        text-align: center;
    }
    .description {
        font-size: 14px;  
        line-height: 1.6;  
        color: #333333;
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;  
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Display the banner across the top
st.markdown('<div class="banner">Spectra Visualization Tool</div>', unsafe_allow_html=True)

# ---------------------------
# User Authentication
# ---------------------------

# Initialize session state variables
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None

if 'plot_sonogram' not in st.session_state:
    st.session_state['plot_sonogram'] = False

if 'peak_finding_enabled' not in st.session_state:
    st.session_state['peak_finding_enabled'] = False

if 'num_peaks' not in st.session_state:
    st.session_state['num_peaks'] = 5

# Simplified authentication: Only username required
st.sidebar.title("User Login")
username = st.sidebar.text_input("Username")
login_button = st.sidebar.button("Login")

if login_button and username:
    user_id = str(uuid.uuid4())  # Assign a unique ID for each user session
    st.session_state['user_id'] = user_id
    st.sidebar.success(f"Logged in as {username}")
elif st.session_state['user_id'] is None:
    st.sidebar.error("Please enter a username and click 'Login' to use the app.")
    st.stop()

user_id = st.session_state['user_id']

# Initialize session state for functional groups based on user (for Peak Detection labels)
functional_groups_key = f'{user_id}_functional_groups'
if functional_groups_key not in st.session_state:
    st.session_state[functional_groups_key] = []

# ---------------------------
# Sidebar Instructions
# ---------------------------

# Move instructions to the sidebar with improved design
st.sidebar.markdown("""
    <div class="sidebar-title">Welcome to the Spectra Visualization Tool</div>
    
    <div class="description">  
    <h3>App Functionalities:</h3>
    <ul>
        <li><b>Data Loading:</b> Use the pre-loaded dataset or upload your own CSV/ZIP file containing molecular spectra data.</li>
        <li><b>SMARTS Filtering:</b> Filter molecules based on structural properties using SMARTS patterns.</li>
        <li><b>Advanced Bond Filtering:</b> Refine your dataset by selecting specific bond types (e.g., C-H, O-H).</li>
        <li><b>Background Opacity:</b> Adjust the transparency of background molecules to emphasize foreground data.</li>
        <li><b>Binning Options:</b> Simplify your spectra by binning data points based on wavelength.</li>
        <li><b>Peak Detection:</b> Enable and configure peak detection parameters to identify significant spectral features.</li>
        <li><b>Functional Group Labels:</b> Add labels to specific wavelengths to identify background gases.</li>
        <li><b>Q-Branch Removal:</b> Remove massive peaks from the spectra for cleaner visualization.</li>
        <li><b>Sonogram Plot:</b> Generate a comprehensive sonogram plot to visualize spectral differences across compounds.</li>
    </ul>
    
    <h3>How to Use:</h3>
    <ol>
        <li><b>Login:</b> Enter your username and click "Login" to access the app.</li>
        <li><b>Data Loading:</b> Choose to use the pre-loaded dataset or upload your own data.</li>
        <li><b>Filtering:</b> Apply SMARTS and/or bond filtering to refine your dataset.</li>
        <li><b>Adjust Opacity:</b> Set the opacity level for background molecules.</li>
        <li><b>Binning:</b> Choose binning options to simplify your spectra visualization.</li>
        <li><b>Peak Detection:</b> Enable peak detection and configure parameters for accurate feature identification.</li>
        <li><b>Functional Group Labels:</b> Add labels to specific wavelengths for easier identification of background gases.</li>
        <li><b>Q-Branch Removal:</b> Specify the wavelength and range of massive peaks to remove them from the spectra.</li>
        <li><b>Plotting:</b> Select foreground molecules and confirm to generate the plots.</li>
        <li><b>Download:</b> After plotting, download the visualizations as PNG files if desired.</li>
    </ol>
    </div>
""", unsafe_allow_html=True)

# ---------------------------
# Data Loading
# ---------------------------

# Preloaded zip URL
ZIP_URL = 'https://raw.githubusercontent.com/praneelshah07/MIT-Project/main/ASM_Vapor_Spectra.csv.zip'

@st.cache_data
def load_data_from_zip(zip_url):
    try:
        response = requests.get(zip_url)
        if response.status_code != 200:
            st.error("Error downloading the zip file from the server.")
            return None

        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        csv_file = None
        for file in zip_file.namelist():
            if file.endswith('.csv'):
                csv_file = file
                break

        if csv_file:
            with zip_file.open(csv_file) as f:
                df = pd.read_csv(f, usecols=["Formula", "IUPAC chemical name", "SMILES", "Molecular Weight", "Boiling Point (oC)", "Raw_Spectra_Intensity"])
            return df
        else:
            st.error("No CSV file found inside the ZIP from the server.")
            return None
    except Exception as e:
        st.error(f"Error extracting CSV from ZIP: {e}")
        return None

# ---------------------------
# Spectra Processing Functions
# ---------------------------

def bin_and_normalize_spectra(spectra, bin_size=None, bin_type='none', q_branch_threshold=0.3, max_peak_limit=0.7, debug=False):
    """
    Function to bin and normalize spectra, with enhanced Q-branch normalization.
    Q-branch normalization is always applied, irrespective of binning.
    
    Parameters:
    - spectra: Raw spectral intensity data (numpy array).
    - bin_size: Size of each bin for binning. If None, no binning is performed.
    - bin_type: Type of binning ('wavelength' or 'none').
    - q_branch_threshold: Threshold for peak detection in Q-branch normalization.
    - max_peak_limit: Maximum allowed intensity for peaks after normalization.
    - debug: If True, plots peak detection for debugging purposes.
    
    Returns:
    - normalized_spectra: The normalized spectral data.
    - x_axis: The corresponding wavelength axis.
    """
    # Define wavenumber range
    wavenumber = np.arange(4000, 500, -1)
    wavelength = 10000 / wavenumber  # Convert wavenumber to wavelength (µm)
    
    # Binning based on bin_type
    if bin_type.lower() == 'wavelength' and bin_size is not None:
        bins = np.arange(wavelength.min(), wavelength.max() + bin_size, bin_size)
        digitized = np.digitize(wavelength, bins)
        x_axis = bins[:-1] + bin_size / 2  # Center of bins

        # Perform binning by averaging spectra in each bin
        binned_spectra = np.array([
            np.mean(spectra[digitized == i]) if np.any(digitized == i) else 0 
            for i in range(1, len(bins))
        ])
    else:
        # No binning; use original spectra
        binned_spectra = spectra.copy()
        x_axis = wavelength

    # Automatic Q-branch normalization
    # Find the highest peak in the spectrum
    highest_peak_idx = np.argmax(binned_spectra)
    highest_peak_intensity = binned_spectra[highest_peak_idx]

    if highest_peak_intensity == 0:
        st.warning("Highest peak intensity is zero. Unable to normalize the spectrum.")
        normalized_spectra = binned_spectra.copy()
    else:
        normalized_spectra = binned_spectra / highest_peak_intensity

    if debug:
        fig_debug, ax_debug = plt.subplots(figsize=(10, 4))
        ax_debug.plot(x_axis, binned_spectra, label='Binned Spectra' if bin_size else 'Original Spectra')
        ax_debug.plot(x_axis[highest_peak_idx], binned_spectra[highest_peak_idx], "x", label='Highest Peak')
        ax_debug.set_title("Q-Branch Normalization")
        ax_debug.set_xlabel("Wavelength (µm)")
        ax_debug.set_ylabel("Intensity")
        ax_debug.legend()
        st.pyplot(fig_debug)
        plt.close(fig_debug)

    return normalized_spectra, x_axis

@st.cache_data
def filter_molecules_by_functional_group(smiles_list, functional_group_smarts):
    """
    Filters molecules based on a SMARTS pattern.

    Parameters:
    - smiles_list: List of SMILES strings.
    - functional_group_smarts: SMARTS pattern to filter molecules.

    Returns:
    - filtered_smiles: List of SMILES strings that match the SMARTS pattern.
    """
    filtered_smiles = []
    try:
        fg_mol = Chem.MolFromSmarts(functional_group_smarts)
        if fg_mol is None:
            st.error("Invalid SMARTS pattern provided.")
            return filtered_smiles
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Add explicit hydrogens to the molecule
                mol = Chem.AddHs(mol)
                # Sanitize molecule to calculate implicit valence
                Chem.SanitizeMol(mol)
                if mol.HasSubstructMatch(fg_mol):
                    filtered_smiles.append(smiles)
    except Exception as e:
        st.error(f"Error in SMARTS filtering: {e}")
    return filtered_smiles

@st.cache_data
def advanced_filtering_by_bond(smiles_list, bond_pattern):
    """
    Filters molecules based on specific bond patterns.
    
    Parameters:
    - smiles_list: List of SMILES strings.
    - bond_pattern: Bond pattern to filter molecules (e.g., "C-H", "C=C").
    
    Returns:
    - filtered_smiles: List of SMILES strings that match the bond pattern.
    """
    bond_patterns = {
        "C-H": "[CH]",
        "C=C": "[C]=[C]",
        "C#C": "[C]#[C]",
        "O-H": "[OH]",
        "N-H": "[NH]",
        "C=O": "[C]=[O]",
        "C-O": "[C][O]",
        "C#N": "[C]#[N]",
        "S-H": "[SH]",
        "N=N": "[N]=[N]",
        "C-S": "[C][S]",
        "C=N": "[C]=[N]",
        "P-H": "[PH]",
        "N-O": "[N][O]",
        "C-Cl": "[C][Cl]",
        "C-Br": "[C][Br]",
        "C-I": "[C][I]",
        "C-F": "[C][F]",
        "O=C-O": "[O][C]=[O]",
        "N-H-N": "[NH][NH]",
        "C-C": "[C][C]",
        "C-N": "[C][N]",
        "C-Si": "[C][Si]",
        "C-P": "[C][P]",
        "C-B": "[C][B]"
    }

    bond_smarts = bond_patterns.get(bond_pattern, bond_pattern)

    try:
        bond_mol = Chem.MolFromSmarts(bond_smarts)
        if bond_mol is None:
            st.error("Invalid bond pattern provided.")
            return []
    except Exception as e:
        st.error(f"Error in bond pattern: {e}")
        return []

    filtered_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Add explicit hydrogens to the molecule
            mol = Chem.AddHs(mol)
            # Sanitize molecule to calculate implicit valence
            Chem.SanitizeMol(mol)
            if mol.HasSubstructMatch(bond_mol):
                filtered_smiles.append(smiles)
    return filtered_smiles

def compute_serial_matrix(dist_mat, method="ward"):
    """
    Performs hierarchical clustering on the distance matrix and orders it.
    
    Parameters:
    - dist_mat: Pairwise distance matrix.
    - method: Linkage method for hierarchical clustering.
    
    Returns:
    - ordered_dist_mat: Reordered distance matrix.
    - res_order: Order of the leaves after clustering.
    - res_linkage: Linkage matrix from clustering.
    """
    if dist_mat.shape[0] < 2:
        raise ValueError("Not enough data for clustering. Ensure at least two molecules are present.")
    
    res_linkage = linkage(dist_mat, method=method)
    res_order = leaves_list(res_linkage)  # Correct order of leaves

    # Reorder distance matrix based on hierarchical clustering leaves
    ordered_dist_mat = dist_mat[res_order, :][:, res_order]
    return ordered_dist_mat, res_order, res_linkage

def detect_peaks_in_range(spectra, x_axis, sensitivity=0.5, max_peaks=5):
    """
    Detect peaks in a specified range of the spectrum.

    Parameters:
    - spectra: Normalized spectral intensity data (numpy array).
    - x_axis: Wavelength axis corresponding to the spectra.
    - sensitivity: Determines the minimum height and prominence of peaks (float between 0 and 1).
    - max_peaks: Maximum number of peaks to detect.

    Returns:
    - peaks: Indices of detected peaks.
    - properties: Properties of the detected peaks.
    """
    # Detect peaks
    peaks, properties = find_peaks(spectra, height=sensitivity, prominence=sensitivity)

    # If more peaks are detected than max_peaks, select the top ones based on prominence
    if len(peaks) > max_peaks:
        prominences = properties['prominences']
        sorted_indices = np.argsort(prominences)[::-1]
        peaks = peaks[sorted_indices[:max_peaks]]
        properties = {key: val[sorted_indices[:max_peaks]] for key, val in properties.items()}

    return peaks, properties

def remove_peak_by_interpolation(spectra, x_axis, peak_wavelength, search_range=0.5):
    """
    Removes a peak from the spectrum by linear interpolation within a specified range around the peak wavelength.

    Parameters:
    - spectra: Normalized spectral intensity data (numpy array).
    - x_axis: Wavelength axis corresponding to the spectra.
    - peak_wavelength: The central wavelength of the peak to remove.
    - search_range: The range (µm) around the peak_wavelength to search for the peak.

    Returns:
    - modified_spectra: The spectra with the peak removed and re-normalized.
    """
    # Define the search window
    lower_bound = peak_wavelength - search_range
    upper_bound = peak_wavelength + search_range

    # Ensure bounds are within the x_axis range
    lower_bound = max(x_axis.min(), lower_bound)
    upper_bound = min(x_axis.max(), upper_bound)

    # Find indices within the search range
    within_range = np.where((x_axis >= lower_bound) & (x_axis <= upper_bound))[0]

    if len(within_range) == 0:
        st.warning("Specified peak wavelength range is out of bounds.")
        return spectra.copy()

    # Extract the sub-spectrum within the range
    sub_spectra = spectra[within_range]
    sub_x_axis = x_axis[within_range]

    # Identify the peak within the range
    peak_idx_within = np.argmax(sub_spectra)
    peak_idx = within_range[peak_idx_within]

    # Define the region to remove (e.g., ±search_range/2)
    # Alternatively, you can define a fixed window around the peak
    removal_width = int(len(sub_spectra) * 0.1)  # Remove 10% of the sub-spectrum around the peak
    start_idx = max(within_range[0], peak_idx - removal_width)
    end_idx = min(within_range[-1], peak_idx + removal_width)

    # Perform linear interpolation to remove the peak
    modified_spectra = spectra.copy()
    # Points before and after the removal region
    if start_idx == 0:
        left_point = spectra[start_idx]
    else:
        left_point = spectra[start_idx - 1]
    if end_idx == len(spectra) - 1:
        right_point = spectra[end_idx]
    else:
        right_point = spectra[end_idx + 1]

    # Linearly interpolate between left_point and right_point over the removal region
    num_points = end_idx - start_idx + 1
    interpolated_values = np.linspace(left_point, right_point, num=num_points)
    modified_spectra[start_idx:end_idx + 1] = interpolated_values

    # Re-normalize the entire spectrum to have a maximum of 1.0
    max_intensity = np.max(modified_spectra)
    if max_intensity > 0:
        modified_spectra = modified_spectra / max_intensity
    else:
        st.warning("After peak removal, the spectrum has zero intensity. Unable to normalize.")

    # Debugging: Display before and after modification for a sample
    # Comment out in production
    # st.write(f"Peak removed between indices {start_idx} and {end_idx}")
    # st.write(f"Left Point: {left_point}, Right Point: {right_point}")

    return modified_spectra

# ---------------------------
# Streamlit Layout
# ---------------------------

# Set up two-column layout below the banner
col1, main_col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="header">Input Controls</div>', unsafe_allow_html=True)

    # Load preloaded data from ZIP
    data = load_data_from_zip(ZIP_URL)
    if data is not None:
        st.success("Using preloaded data from GitHub zip file.")

    # File uploader for custom datasets
    uploaded_file = st.file_uploader("Upload your dataset (CSV or ZIP):", type=["csv", "zip"])

    # Load new dataset if uploaded
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as z:
                file_list = z.namelist()
                csv_file = None
                for file in file_list:
                    if file.endswith('.csv'):
                        csv_file = file
                        break
                if csv_file:
                    with z.open(csv_file) as f:
                        try:
                            data = pd.read_csv(f, usecols=["Formula", "IUPAC chemical name", "SMILES", "Molecular Weight", "Boiling Point (oC)", "Raw_Spectra_Intensity"])
                            st.success(f"Extracted and loaded: {csv_file}")
                        except Exception as e:
                            st.error(f"Error reading CSV from ZIP: {e}")
                else:
                    st.error("No CSV file found inside the uploaded ZIP.")
        
        elif uploaded_file.name.endswith('.csv'):
            try:
                data = pd.read_csv(uploaded_file, usecols=["Formula", "IUPAC chemical name", "SMILES", "Molecular Weight", "Boiling Point (oC)", "Raw_Spectra_Intensity"])
                st.success("Loaded uploaded CSV file successfully.")
            except pd.errors.EmptyDataError:
                st.error("Uploaded CSV is empty.")
            except KeyError:
                st.error("Uploaded CSV is missing required columns.")
            except Exception as e:
                st.error(f"Error reading uploaded CSV: {e}")

    # Display dataset preview
    if data is not None:
        try:
            # Ensure 'Raw_Spectra_Intensity' is a string before loading JSON
            data['Raw_Spectra_Intensity'] = data['Raw_Spectra_Intensity'].astype(str)
            data['Raw_Spectra_Intensity'] = data['Raw_Spectra_Intensity'].apply(json.loads)
            data['Raw_Spectra_Intensity'] = data['Raw_Spectra_Intensity'].apply(lambda x: np.array(x) if isinstance(x, list) else np.array([]))
            
            # Check for empty arrays
            empty_spectra = data['Raw_Spectra_Intensity'].apply(lambda x: len(x) == 0)
            if empty_spectra.any():
                st.warning(f"{empty_spectra.sum()} entries have empty 'Raw_Spectra_Intensity'. They will be excluded.")
                data = data[~empty_spectra]
        except json.JSONDecodeError as e:
            st.error(f"JSON decoding error in 'Raw_Spectra_Intensity': {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error processing 'Raw_Spectra_Intensity': {e}")
            st.stop()

        # Validate required columns
        required_columns = ["Formula", "IUPAC chemical name", "SMILES", "Molecular Weight", "Boiling Point (oC)"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"The following required columns are missing from the dataset: {', '.join(missing_columns)}")
            st.stop()

        st.write(data[required_columns])

        # Initialize filtered_smiles with all unique SMILES
        filtered_smiles = data['SMILES'].unique()

        # Advanced Filtration Metrics
        with st.expander("Advanced Filtration Metrics"):
            # Utilize tabs for better organization
            # Added "Q-Branch Removal" tab
            tabs = st.tabs(["Background Settings", "Binning", "Peak Detection", "Q-Branch Removal"])

            with tabs[0]:
                st.subheader("Background Settings")

                # SMARTS Filtering
                st.markdown("**SMARTS Filtering**")
                functional_group_smarts = st.text_input(
                    "Enter a SMARTS pattern to filter background molecules:",
                    value="",  # Display the input box immediately
                    help="Provide a valid SMARTS pattern to filter molecules that match the specified structural features."
                )
                if functional_group_smarts:
                    filtered_smiles_smarts = filter_molecules_by_functional_group(data['SMILES'].unique(), functional_group_smarts)
                    st.write(f"Filtered dataset to {len(filtered_smiles_smarts)} molecules using SMARTS pattern.")
                    filtered_smiles = np.intersect1d(filtered_smiles, filtered_smiles_smarts)

                # Advanced Bond Filtering
                st.markdown("**Advanced Bond Filtering**")
                bond_input = st.selectbox(
                    "Select a bond type to filter background molecules:", 
                    ["None", "C-H", "C=C", "C#C", "O-H", "N-H", 
                     "C=O", "C-O", "C#N", "S-H", "N=N", "C-S", "C=N", "P-H",
                     "N-O", "C-Cl", "C-Br", "C-I", "C-F", "O=C-O", "N-H-N", "C-C", "C-N", "C-Si", "C-P", "C-B"],
                    help="Choose a specific bond type to filter molecules that contain this bond."
                )
                if bond_input and bond_input != "None":
                    filtered_smiles_bond = advanced_filtering_by_bond(data['SMILES'].unique(), bond_input)
                    st.write(f"Filtered dataset to {len(filtered_smiles_bond)} molecules with bond pattern '{bond_input}'.")
                    filtered_smiles = np.intersect1d(filtered_smiles, filtered_smiles_bond)

                # Background molecule opacity control
                st.markdown("**Background Molecule Opacity**")
                background_opacity = st.slider(
                    'Set Background Molecule Opacity:',
                    min_value=0.0,
                    max_value=1.0,
                    value=0.01,
                    step=0.01,
                    help="Adjust the transparency of background molecules. Lower values make them more transparent."
                )

            with tabs[1]:
                st.subheader("Binning Options")
                bin_type = st.selectbox(
                    'Select binning type:',
                    ['None', 'Wavelength'],
                    help="Choose how to bin the spectral data. Binning by wavelength can simplify the spectra by averaging data points."
                )

                if bin_type == 'Wavelength':
                    bin_size = st.number_input(
                        'Enter bin size (µm):',
                        min_value=0.001,
                        max_value=5.0,
                        value=0.1,
                        step=0.001,
                        format="%.3f",
                        help="Specify the size of each wavelength bin in micrometers (µm). Smaller bin sizes retain more detail."
                    )
                else:
                    bin_size = None

            with tabs[2]:
                st.subheader("Peak Detection")

                st.markdown("""
                Peak detection helps identify significant features in the spectral data. Adjust the sensitivity to control how easily peaks are detected.
                """)

                # Simplified Peak Detection Parameters
                peak_sensitivity = st.slider(
                    "Sensitivity",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Adjust the sensitivity of peak detection. Higher values detect more prominent peaks."
                )

                max_peaks = st.slider(
                    "Maximum Number of Peaks to Display",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1,
                    help="Set the maximum number of peaks to identify and label."
                )

                # Enable Peak Finding and Labeling
                enable_peak_finding = st.checkbox(
                    'Enable Peak Detection and Labeling',
                    value=False,
                    help="Check to enable automatic peak detection and labeling on the spectra plots."
                )

                if enable_peak_finding:
                    st.markdown("### Peak Labels")

                    # Form to input functional group data based on wavelength
                    with st.form(key='functional_group_form'):
                        fg_label = st.text_input(
                            "Functional Group Label (e.g., C-C, N=C=O)",
                            help="Name of the functional group to label the peak."
                        )
                        fg_wavelength = st.number_input(
                            "Wavelength Position (µm)",
                            min_value=3.0,
                            max_value=20.0,
                            value=15.0,
                            step=0.1,
                            help="Wavelength position where the functional group peak is expected."
                        )
                        add_fg = st.form_submit_button("Add Functional Group")

                    if add_fg:
                        if fg_label and fg_wavelength:
                            st.session_state[functional_groups_key].append({
                                'Functional Group': fg_label,
                                'Wavelength': fg_wavelength
                            })
                            st.success(f"Added functional group: {fg_label} at {fg_wavelength} µm")
                        else:
                            st.error("Please provide both label and wavelength for the functional group.")

                    # Display existing functional group labels and allow deletion
                    if st.session_state[functional_groups_key]:
                        st.markdown("**Current Functional Group Labels:**")
                        for i, fg in enumerate(st.session_state[functional_groups_key]):
                            col_label, col_wavelength, col_delete = st.columns([2, 2, 1])
                            with col_label:
                                st.write(f"**Label:** {fg['Functional Group']}")
                            with col_wavelength:
                                st.write(f"**Wavelength:** {fg['Wavelength']} µm")
                            with col_delete:
                                if st.button(f"Delete", key=f"delete_fg_{i}"):
                                    st.session_state[functional_groups_key].pop(i)
                                    st.success(f"Deleted functional group: {fg['Functional Group']}")

            with tabs[3]:
                st.subheader("Q-Branch Removal")

                st.markdown("""
                **Q-Branch Removal** allows you to remove massive peaks from the spectra for cleaner visualization. Specify the central wavelength and the range around it to focus the peak removal process.
                """)

                # User input for Q-Branch Removal
                q_branch_wavelength = st.number_input(
                    "Enter the Central Wavelength of the Massive Peak to Remove (µm):",
                    min_value=3.0,
                    max_value=20.0,
                    value=15.0,
                    step=0.1,
                    help="Specify the central wavelength position of the massive peak you want to remove from the spectra."
                )

                # Additional parameters for peak removal
                search_range = st.number_input(
                    "Enter the Range around the Peak Wavelength to Search (µm):",
                    min_value=0.1,
                    max_value=5.0,
                    value=0.5,
                    step=0.1,
                    help="Define the range around the central wavelength to search for peak boundaries."
                )

                # Button to apply Q-Branch Removal
                apply_q_branch = st.button("Apply Q-Branch Removal")

                if apply_q_branch:
                    # Apply Q-Branch Removal to all spectra
                    def apply_q_branch_removal(row):
                        spectra = row['Raw_Spectra_Intensity']
                        normalized_spectra, x_axis = bin_and_normalize_spectra(
                            spectra, 
                            bin_size=bin_size, 
                            bin_type=bin_type.lower(),  # Now 'wavelength' or 'none'
                            q_branch_threshold=0.3,  # Fixed threshold for automatic normalization
                            max_peak_limit=0.7,
                            debug=False  # Disable debug mode for regular plotting
                        )
                        modified_spectra = remove_peak_by_interpolation(
                            normalized_spectra, 
                            x_axis, 
                            peak_wavelength=q_branch_wavelength, 
                            search_range=search_range
                        )
                        return modified_spectra

                    # Apply the Q-Branch Removal
                    try:
                        data['Modified_Spectra'] = data.apply(apply_q_branch_removal, axis=1)
                        st.success("Q-Branch Removal applied successfully to all relevant spectra.")
                        
                        # Debugging: Show before and after for a sample
                        sample_index = 0
                        sample_smiles = data['SMILES'].iloc[sample_index]
                        st.write(f"**Sample Molecule:** {sample_smiles}")
                        st.write("**Original Spectrum:**")
                        st.write(data['Raw_Spectra_Intensity'].iloc[sample_index])
                        st.write("**Modified Spectrum:**")
                        st.write(data['Modified_Spectra'].iloc[sample_index])
                        
                    except Exception as e:
                        st.error(f"Error applying Q-Branch Removal: {e}")

    with main_col2:
        st.markdown('<div class="header">Graph</div>', unsafe_allow_html=True)

        if confirm_button:
            if data is None:
                st.error("No data available to plot.")
            else:
                # Determine which spectra to plot: Modified or Original
                if 'Modified_Spectra' in data.columns:
                    spectra_column = 'Modified_Spectra'
                    st.info("Using modified spectra with Q-Branch Removal applied.")
                else:
                    spectra_column = 'Raw_Spectra_Intensity'

                if (len(selected_smiles) == 0) and (not st.session_state['plot_sonogram']):
                    st.warning("No foreground molecules selected and Sonogram plotting is disabled.")
                else:
                    with st.spinner('Generating plots, this may take some time...'):
                        # Plotting logic
                        if len(selected_smiles) > 0:
                            fig_spec, ax_spec = plt.subplots(figsize=(16, 6.5), dpi=100)
                            # Define allowed colors for selection
                            allowed_colors_plot = [
                                'red', 'green', 'blue', 'cyan', 'magenta', 'orange',
                                'purple', 'pink', 'brown', 'gray', 'lime', 'maroon',
                                'navy', 'teal', 'olive', 'coral', 'gold', 'indigo',
                                'violet', 'turquoise', 'salmon'
                            ]
                            random.shuffle(allowed_colors_plot)  # Shuffle to provide varied color assignments
                            target_spectra = {}

                            # Automatically select all background molecules if none specified
                            if not filtered_smiles.size:
                                background_smiles = data['SMILES'].unique()
                            else:
                                background_smiles = filtered_smiles

                            # First plot background molecules
                            for smiles in background_smiles:
                                spectra_row = data[data['SMILES'] == smiles]
                                if spectra_row.empty:
                                    continue
                                spectra = spectra_row.iloc[0][spectra_column]
                                # Apply binning and normalization
                                normalized_spectra, x_axis = bin_and_normalize_spectra(
                                    spectra, 
                                    bin_size=bin_size, 
                                    bin_type=bin_type.lower(),  # Now 'wavelength' or 'none'
                                    q_branch_threshold=0.3,  # Fixed threshold for automatic normalization
                                    max_peak_limit=0.7,
                                    debug=False  # Disable debug mode for regular plotting
                                )
                                # Plot background molecule
                                ax_spec.fill_between(x_axis, 0, normalized_spectra, color="k", alpha=background_opacity, label="_nolegend_")

                            # Then plot foreground molecules to ensure they are on top
                            for idx, smiles in enumerate(selected_smiles):
                                spectra_row = data[data['SMILES'] == smiles]
                                if spectra_row.empty:
                                    continue
                                spectra = spectra_row.iloc[0][spectra_column]
                                # Apply binning and normalization
                                normalized_spectra, x_axis = bin_and_normalize_spectra(
                                    spectra, 
                                    bin_size=bin_size, 
                                    bin_type=bin_type.lower(),  # Now 'wavelength' or 'none'
                                    q_branch_threshold=0.3,  # Fixed threshold for automatic normalization
                                    max_peak_limit=0.7,
                                    debug=False  # Disable debug mode for regular plotting
                                )
                                target_spectra[smiles] = normalized_spectra

                                # Get user-selected color for the molecule
                                color = foreground_colors.get(smiles, 'blue')  # Default to blue if not set

                                # Plot foreground molecule
                                ax_spec.fill_between(x_axis, 0, normalized_spectra, color=color, alpha=0.7, label=smiles)

                                if enable_peak_finding:
                                    # Detect peaks with simplified parameters
                                    detected_peaks, detected_properties = detect_peaks_in_range(
                                        normalized_spectra,
                                        x_axis,
                                        sensitivity=peak_sensitivity,
                                        max_peaks=max_peaks
                                    )

                                    # Handle overlapping peaks by creating a set of unique wavelengths
                                    unique_peak_wavelengths = {}
                                    for peak in detected_peaks:
                                        peak_wavelength = x_axis[peak]
                                        # Round wavelength to one decimal to group similar peaks
                                        rounded_wavelength = round(peak_wavelength, 1)
                                        if rounded_wavelength not in unique_peak_wavelengths:
                                            unique_peak_wavelengths[rounded_wavelength] = peak

                                    # Label the detected peaks with yellow background and black text
                                    for rounded_wavelength, peak in unique_peak_wavelengths.items():
                                        peak_wavelength = x_axis[peak]
                                        peak_intensity = normalized_spectra[peak]
                                        ax_spec.plot(peak_wavelength, peak_intensity, "x", color=color)
                                        ax_spec.text(
                                            peak_wavelength,
                                            peak_intensity + 0.02,
                                            f'{peak_wavelength:.1f} µm',
                                            fontsize=9,
                                            ha='center',
                                            color='black',  # Set text color to black
                                            bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none')
                                        )

                            # Add functional group labels for background gases based on wavelength
                            for fg in st.session_state[functional_groups_key]:
                                fg_wavelength = fg['Wavelength']
                                fg_label = fg['Functional Group']
                                ax_spec.axvline(fg_wavelength, color='grey', linestyle='--')
                                ax_spec.text(fg_wavelength, 1, fg_label, fontsize=12, color='black', ha='center')

                            # Customize plot
                            ax_spec.set_xlim([x_axis.min(), x_axis.max()])

                            major_ticks = [3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 20]
                            ax_spec.set_xticks(major_ticks)
                            ax_spec.set_xticklabels([str(tick) for tick in major_ticks])

                            ax_spec.tick_params(direction="in",
                                labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                                bottom=True, top=True, left=True, right=True)

                            ax_spec.set_xlabel("Wavelength ($\mu$m)", fontsize=22)
                            ax_spec.set_ylabel("Absorbance (Normalized to 1)", fontsize=22)

                            if selected_smiles:
                                ax_spec.legend()

                            st.pyplot(fig_spec)

                            # Download button for the spectra plot
                            buf_spec = io.BytesIO()
                            fig_spec.savefig(buf_spec, format='png')
                            buf_spec.seek(0)
                            st.download_button(
                                label="Download Plot as PNG",
                                data=buf_spec,
                                file_name="spectra_plot.png",
                                mime="image/png"
                            )
                            plt.close(fig_spec)

                            # Optional: Before-and-After Plot for a Sample Molecule
                            if 'Modified_Spectra' in data.columns and len(selected_smiles) > 0:
                                sample_smiles = selected_smiles[0]
                                original_spectra = data[data['SMILES'] == sample_smiles]['Raw_Spectra_Intensity'].values[0]
                                modified_spectra = data[data['SMILES'] == sample_smiles]['Modified_Spectra'].values[0]
                                normalized_original, x_axis_original = bin_and_normalize_spectra(
                                    original_spectra,
                                    bin_size=bin_size,
                                    bin_type=bin_type.lower(),
                                    q_branch_threshold=0.3,
                                    max_peak_limit=0.7,
                                    debug=False
                                )
                                normalized_modified, x_axis_modified = bin_and_normalize_spectra(
                                    modified_spectra,
                                    bin_size=bin_size,
                                    bin_type=bin_type.lower(),
                                    q_branch_threshold=0.3,
                                    max_peak_limit=0.7,
                                    debug=False
                                )

                                fig_before_after, ax_before_after = plt.subplots(figsize=(12, 6))
                                ax_before_after.plot(x_axis_original, normalized_original, label='Original Spectra', alpha=0.7)
                                ax_before_after.plot(x_axis_modified, normalized_modified, label='Modified Spectra', alpha=0.7)
                                ax_before_after.set_title(f"Before and After Q-Branch Removal for {sample_smiles}")
                                ax_before_after.set_xlabel("Wavelength ($\mu$m)")
                                ax_before_after.set_ylabel("Absorbance (Normalized to 1)")
                                ax_before_after.legend()
                                st.pyplot(fig_before_after)
                                plt.close(fig_before_after)
