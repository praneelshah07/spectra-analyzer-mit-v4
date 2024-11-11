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
from scipy.ndimage import gaussian_filter1d  # For optional smoothing

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

# Initialize session state for Q-branch removals
if 'q_branch_removals' not in st.session_state:
    st.session_state['q_branch_removals'] = []

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

def split_segments(x, y):
    """
    Split the data into continuous segments where y is not NaN.

    Parameters:
    - x: numpy array of x-axis values.
    - y: numpy array of y-axis values.

    Returns:
    - List of (x_segment, y_segment) tuples.
    """
    segments = []
    current_segment_x = []
    current_segment_y = []

    for xi, yi in zip(x, y):
        if not np.isnan(yi):
            current_segment_x.append(xi)
            current_segment_y.append(yi)
        else:
            if current_segment_x:
                segments.append((np.array(current_segment_x), np.array(current_segment_y)))
                current_segment_x = []
                current_segment_y = []
    if current_segment_x:
        segments.append((np.array(current_segment_x), np.array(current_segment_y)))
    return segments

def bin_and_normalize_spectra(
    spectra, 
    x_axis, 
    bin_size=None, 
    bin_type='none', 
    q_branch_threshold=0.3, 
    max_peak_limit=0.7, 
    q_branch_removals=None,  # New parameter
    interpolate=True,        # New parameter to toggle interpolation
    debug=False
):
    """
    Function to bin and normalize spectra, with Q-branch removal, renormalization, and interpolation.

    Parameters:
    - spectra: Raw spectral intensity data (numpy array).
    - x_axis: Wavelength axis corresponding to spectra (numpy array).
    - bin_size: Size of each bin for binning. If None, no binning is performed.
    - bin_type: Type of binning ('wavelength' or 'none').
    - q_branch_threshold: Threshold for peak detection in Q-branch normalization.
    - max_peak_limit: Maximum allowed intensity for peaks after normalization.
    - q_branch_removals: List of dictionaries with 'start' and 'end' wavelengths to remove.
    - interpolate: Boolean indicating whether to interpolate NaN regions.
    - debug: If True, plots peak detection for debugging purposes.

    Returns:
    - normalized_spectra: The normalized spectral data with Q-branches removed, interpolated, and renormalized to 1.
    - x_axis_binned: The corresponding wavelength axis after binning.
    - peaks: Indices of detected peaks.
    - properties: Properties of the detected peaks.
    """
    # Binning based on bin_type
    if bin_type.lower() == 'wavelength' and bin_size is not None:
        # Corrected binning to prevent extra bin and ensure matching lengths
        bins = np.arange(x_axis.min(), x_axis.max() + bin_size, bin_size)  # Include the last bin edge
        digitized = np.digitize(x_axis, bins) - 1  # Adjust digitize to start from 0
        x_axis_binned = bins[:-1] + bin_size / 2  # Centers of bins

        # Perform binning by averaging spectra in each bin
        binned_spectra = np.array([
            np.mean(spectra[digitized == i]) if np.any(digitized == i) else 0  # Set empty bins to 0
            for i in range(len(bins) - 1)  # Number of bins is len(bins) -1
        ])
    else:
        # No binning; use original spectra
        binned_spectra = spectra.copy()
        x_axis_binned = x_axis

    # Automatic Q-branch normalization
    # Find the highest peak in the spectrum, ignoring NaNs
    if np.all(np.isnan(binned_spectra)):
        st.warning("All spectral data has been removed due to Q-branch removal. Unable to normalize the spectrum.")
        normalized_spectra = binned_spectra.copy()
        peaks = []
        properties = {}
        return normalized_spectra, x_axis_binned, peaks, properties

    highest_peak_idx = np.nanargmax(binned_spectra)
    highest_peak_intensity = binned_spectra[highest_peak_idx]

    if highest_peak_intensity == 0 or np.isnan(highest_peak_intensity):
        st.warning("Highest peak intensity is zero or NaN. Unable to normalize the spectrum.")
        normalized_spectra = binned_spectra.copy()
    else:
        normalized_spectra = binned_spectra / highest_peak_intensity

    # Remove Q-branch regions by setting to NaN
    if q_branch_removals:
        for removal in q_branch_removals:
            start = removal['start']
            end = removal['end']
            mask = ~((x_axis_binned >= start) & (x_axis_binned <= end))
            # Set the y-values in the removal ranges to NaN
            normalized_spectra = np.where(mask, normalized_spectra, np.nan)

    # Renormalize after Q-branch removal
    if q_branch_removals:
        # Find the new maximum excluding NaNs
        if np.all(np.isnan(normalized_spectra)):
            st.warning("All spectral data has been removed after Q-branch removal. Unable to renormalize.")
            # Optionally, set all to 0 or handle as needed
        else:
            new_max = np.nanmax(normalized_spectra)
            if new_max > 0:
                normalized_spectra = normalized_spectra / new_max
            else:
                st.warning("New maximum after Q-branch removal is zero. Unable to renormalize.")

    # Interpolate NaN regions to smooth gaps
    if interpolate:
        # Convert to pandas Series for interpolation
        spectra_series = pd.Series(normalized_spectra, index=x_axis_binned)
        # Perform linear interpolation
        spectra_interpolated = spectra_series.interpolate(method='linear', limit_direction='both')
        # Update normalized_spectra
        normalized_spectra = spectra_interpolated.values

        if debug:
            fig_interp, ax_interp = plt.subplots(figsize=(10, 4))
            ax_interp.plot(x_axis_binned, normalized_spectra, label='Interpolated Spectra')
            ax_interp.set_title("Spectra After Interpolation")
            ax_interp.set_xlabel("Wavelength (µm)")
            ax_interp.set_ylabel("Normalized Intensity")
            ax_interp.legend()
            st.pyplot(fig_interp)
            plt.close(fig_interp)

    # Detect peaks after normalization and interpolation
    peaks, properties = detect_peaks(normalized_spectra, sensitivity=q_branch_threshold, max_peaks=int(max_peak_limit * 10))  # Adjust max_peaks as needed

    if debug:
        fig_debug, ax_debug = plt.subplots(figsize=(10, 4))
        ax_debug.plot(x_axis_binned, normalized_spectra, label='Binned, Normalized, and Interpolated Spectra' if bin_size else 'Original Spectra')
        if not np.isnan(normalized_spectra).all():
            ax_debug.plot(x_axis_binned[highest_peak_idx], normalized_spectra[highest_peak_idx], "x", label='Highest Peak')
        ax_debug.set_title("Q-Branch Normalization, Renormalization, and Interpolation")
        ax_debug.set_xlabel("Wavelength (µm)")
        ax_debug.set_ylabel("Normalized Intensity")
        ax_debug.legend()
        st.pyplot(fig_debug)
        plt.close(fig_debug)

    return normalized_spectra, x_axis_binned, peaks, properties

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

# ---------------------------
# Peak Detection Function
# ---------------------------

def detect_peaks(spectra, sensitivity=0.5, max_peaks=5):
    """
    Detect peaks in a spectrum using simplified parameters.

    Parameters:
    - spectra: Normalized spectral intensity data (numpy array).
    - sensitivity: Determines the minimum height and prominence of peaks (float between 0 and 1).
    - max_peaks: Maximum number of peaks to detect.

    Returns:
    - peaks: Indices of detected peaks.
    - properties: Properties of the detected peaks.
    """
    # Adjust height and prominence based on sensitivity
    height = sensitivity
    prominence = sensitivity

    # Detect peaks
    peaks, properties = find_peaks(spectra, height=height, prominence=prominence)

    # If more peaks are detected than max_peaks, select the top ones based on prominence
    if len(peaks) > max_peaks:
        prominences = properties['prominences']
        sorted_indices = np.argsort(prominences)[::-1]
        peaks = peaks[sorted_indices[:max_peaks]]
        properties = {key: val[sorted_indices[:max_peaks]] for key, val in properties.items()}

    return peaks, properties

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
            tabs = st.tabs(["Background Settings", "Binning", "Peak Detection", "Q-Branch Removal"])

            # Background Settings Tab
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

            # Binning Options Tab
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

            # Peak Detection Tab
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

            # Q-Branch Removal Tab
            with tabs[3]:
                st.subheader("Q-Branch Removal")

                # Checkbox to enable Q-branch removal
                enable_q_branch = st.checkbox(
                    "Enable Q-Branch Removal",
                    value=False,
                    help="Check to enable removal of specific Q-branch peaks from foreground spectra."
                )

                if enable_q_branch:
                    st.markdown("""
                    Removing the Q-branch helps in eliminating specific peaks that may interfere with your analysis.
                    Specify the wavelength range of the Q-branch you want to remove.
                    """)

                    # Form to input Q-branch removal ranges
                    with st.form(key='q_branch_removal_form'):
                        q_start = st.number_input(
                            "Start Wavelength (µm)",
                            min_value=2.5,  # x_axis_min = 10000 / 4000 = 2.5 µm
                            max_value=20.0,  # x_axis_max = 10000 / 500 = 20 µm
                            value=15.0,
                            step=0.1,
                            help="Enter the starting wavelength of the Q-branch to remove."
                        )
                        q_end = st.number_input(
                            "End Wavelength (µm)",
                            min_value=2.5,
                            max_value=20.0,
                            value=16.0,
                            step=0.1,
                            help="Enter the ending wavelength of the Q-branch to remove."
                        )
                        add_q_branch = st.form_submit_button("Remove Q-Branch")

                    if add_q_branch:
                        if q_start >= q_end:
                            st.error("Start wavelength must be less than end wavelength.")
                        else:
                            # Check for overlapping ranges
                            overlap = False
                            for removal in st.session_state['q_branch_removals']:
                                if not (q_end < removal['start'] or q_start > removal['end']):
                                    overlap = True
                                    break
                            if overlap:
                                st.error("The specified range overlaps with an existing Q-branch removal.")
                            else:
                                st.session_state['q_branch_removals'].append({'start': q_start, 'end': q_end})
                                st.success(f"Removed Q-branch from {q_start} µm to {q_end} µm.")

                    # Display current Q-branch removals
                    if st.session_state['q_branch_removals']:
                        st.markdown("**Current Q-Branch Removals:**")
                        for i, q in enumerate(st.session_state['q_branch_removals']):
                            col_start, col_end, col_delete = st.columns([2, 2, 1])
                            with col_start:
                                st.write(f"**Start:** {q['start']} µm")
                            with col_end:
                                st.write(f"**End:** {q['end']} µm")
                            with col_delete:
                                if st.button(f"Delete", key=f"delete_q_branch_{i}"):
                                    st.session_state['q_branch_removals'].pop(i)
                                    st.success(f"Deleted Q-branch removal from {q['start']} µm to {q['end']} µm.")

    # Foreground Molecules Selection (Clean Interface)
    selected_smiles = st.multiselect('Select Foreground Molecules:', data['SMILES'].unique())

    # Color Selection for Foreground Molecules
    if selected_smiles:
        st.markdown("**Select Colors for Foreground Molecules**")
        foreground_colors = {}
        # Define allowed colors excluding black and yellow
        allowed_colors = [
            'red', 'green', 'blue', 'cyan', 'magenta', 'orange',
            'purple', 'pink', 'brown', 'gray', 'lime', 'maroon',
            'navy', 'teal', 'olive', 'coral', 'gold', 'indigo',
            'violet', 'turquoise', 'salmon'
        ]
        for idx, smiles in enumerate(selected_smiles):
            # Assign default color based on index to ensure different defaults
            default_color = allowed_colors[idx % len(allowed_colors)]
            # Find the index of the default color to set it as the selected option
            default_color_index = allowed_colors.index(default_color)
            foreground_colors[smiles] = st.selectbox(
                f"Select color for molecule {smiles}",
                options=allowed_colors,
                index=default_color_index,
                key=f"color_{smiles}"
            )
    else:
        foreground_colors = {}

    # Confirm button
    confirm_button = st.button('Confirm Selection and Start Plotting')

with main_col2:
    st.markdown('<div class="header">Graph</div>', unsafe_allow_html=True)

    if confirm_button:
        if data is None:
            st.error("No data available to plot.")
        else:
            if (len(selected_smiles) == 0) and (not st.session_state['plot_sonogram']):
                st.warning("No foreground molecules selected and Sonogram plotting is disabled.")
            else:
                with st.spinner('Generating plots, this may take some time...'):
                    if st.session_state['plot_sonogram']:
                        # Sonogram plotting logic
                        intensity_data = np.array(data[data['SMILES'].isin(filtered_smiles)]['Raw_Spectra_Intensity'].tolist())
                        if len(intensity_data) > 1:
                            try:
                                dist_mat = squareform(pdist(intensity_data))
                                ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, "ward")

                                fig_sono, ax_sono = plt.subplots(figsize=(12, 12))
                                ax_sono.imshow(np.array(intensity_data)[res_order], aspect='auto', extent=[2.5, 20, len(ordered_dist_mat), 0], cmap='viridis')
                                ax_sono.set_xlabel("Wavelength (µm)")
                                ax_sono.set_ylabel("Molecules")
                                ax_sono.set_title("Sonogram Plot")

                                st.pyplot(fig_sono)
                                plt.close(fig_sono)

                                # Download button for the sonogram
                                buf_sono = io.BytesIO()
                                fig_sono.savefig(buf_sono, format='png')
                                buf_sono.seek(0)
                                st.download_button(label="Download Sonogram as PNG", data=buf_sono, file_name="sonogram.png", mime="image/png")
                            except Exception as e:
                                st.error(f"Error generating sonogram: {e}")
                        else:
                            st.error("Not enough data to generate the sonogram. Please ensure there are at least two molecules.")

                    # Spectra plotting logic
                    if len(selected_smiles) > 0:
                        fig_spec, ax_spec = plt.subplots(figsize=(16, 6.5), dpi=100)
                        # Define allowed colors for selection
                        allowed_colors = [
                            'red', 'green', 'blue', 'cyan', 'magenta', 'orange',
                            'purple', 'pink', 'brown', 'gray', 'lime', 'maroon',
                            'navy', 'teal', 'olive', 'coral', 'gold', 'indigo',
                            'violet', 'turquoise', 'salmon'
                        ]
                        random.shuffle(allowed_colors)  # Shuffle to provide varied color assignments
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
                            spectra = spectra_row.iloc[0]['Raw_Spectra_Intensity']
                            # Define x_axis based on the length of spectra
                            # Correct the wavenumber range to match spectra length
                            # Assuming 'Raw_Spectra_Intensity' has 3500 points
                            wavenumber = np.arange(4000, 500, -1)  # 4000 to 501 inclusive, step=-1
                            x_axis = 10000 / wavenumber  # Convert wavenumber to wavelength (µm)
                            # Ensure x_axis and spectra have the same length
                            if len(x_axis) != len(spectra):
                                st.error(f"Length mismatch for molecule {smiles}: x_axis ({len(x_axis)}) vs spectra ({len(spectra)}). Skipping plotting for this molecule.")
                                continue
                            # Apply binning and normalization with interpolation disabled for background
                            normalized_spectra, x_axis_binned, _, _ = bin_and_normalize_spectra(
                                spectra, 
                                x_axis=x_axis,
                                bin_size=bin_size, 
                                bin_type=bin_type.lower(),  # Now 'wavelength' or 'none'
                                q_branch_threshold=0.3,  # Fixed threshold for automatic normalization
                                max_peak_limit=0.7,
                                q_branch_removals=None,  # Do not remove Q-branch from background
                                interpolate=False,        # Disable interpolation for background
                                debug=False  # Disable debug mode for regular plotting
                            )
                            # Data Validation Before Plotting
                            if len(x_axis_binned) != len(normalized_spectra):
                                st.error(f"Mismatch in lengths: x_axis_binned ({len(x_axis_binned)}) vs normalized_spectra ({len(normalized_spectra)}) for molecule {smiles}.")
                                continue
                            else:
                                if np.any(np.isnan(normalized_spectra)) or np.any(np.isinf(normalized_spectra)):
                                    st.warning(f"Spectral data contains NaN or Inf values for molecule {smiles}. These will be set to 0 for plotting.")
                                    normalized_spectra = np.nan_to_num(normalized_spectra, nan=0.0, posinf=0.0, neginf=0.0)
                                ax_spec.fill_between(x_axis_binned, 0, normalized_spectra, color="k", alpha=background_opacity)

                        # Then plot foreground molecules to ensure they are on top
                        for idx, smiles in enumerate(selected_smiles):
                            spectra_row = data[data['SMILES'] == smiles]
                            if spectra_row.empty:
                                continue
                            spectra = spectra_row.iloc[0]['Raw_Spectra_Intensity']
                            # Define x_axis based on the length of spectra
                            # Correct the wavenumber range to match spectra length
                            # Assuming 'Raw_Spectra_Intensity' has 3500 points
                            wavenumber = np.arange(4000, 500, -1)  # 4000 to 501 inclusive, step=-1
                            x_axis = 10000 / wavenumber  # Convert wavenumber to wavelength (µm)
                            # Ensure x_axis and spectra have the same length
                            if len(x_axis) != len(spectra):
                                st.error(f"Length mismatch for molecule {smiles}: x_axis ({len(x_axis)}) vs spectra ({len(spectra)}). Skipping plotting for this molecule.")
                                continue
                            # Apply binning and normalization, including Q-branch removals if enabled
                            if st.session_state.get('q_branch_removals') and enable_q_branch:
                                normalized_spectra, x_axis_binned, _, _ = bin_and_normalize_spectra(
                                    spectra, 
                                    x_axis=x_axis,
                                    bin_size=bin_size, 
                                    bin_type=bin_type.lower(),  # Now 'wavelength' or 'none'
                                    q_branch_threshold=0.3,  # Fixed threshold for automatic normalization
                                    max_peak_limit=0.7,
                                    q_branch_removals=st.session_state['q_branch_removals'],  # Pass the removals
                                    interpolate=True,         # Enable interpolation for foreground
                                    debug=False  # Disable debug mode for regular plotting
                                )
                            else:
                                normalized_spectra, x_axis_binned, _, _ = bin_and_normalize_spectra(
                                    spectra, 
                                    x_axis=x_axis,
                                    bin_size=bin_size, 
                                    bin_type=bin_type.lower(),
                                    q_branch_threshold=0.3,
                                    max_peak_limit=0.7,
                                    q_branch_removals=None,
                                    interpolate=True,        # Enable interpolation even if Q-branch removal is not enabled
                                    debug=False
                                )
                            target_spectra[smiles] = normalized_spectra

                            # Get user-selected color for the molecule
                            color = foreground_colors.get(smiles, 'blue')  # Default to blue if not set

                            # Split the data into segments excluding Q-branch ranges
                            if st.session_state.get('q_branch_removals') and enable_q_branch:
                                segments = split_segments(x_axis_binned, normalized_spectra)
                            else:
                                segments = [(x_axis_binned, normalized_spectra)]

                            # Plot each segment separately to create gaps
                            for segment_x, segment_y in segments:
                                ax_spec.fill_between(segment_x, 0, segment_y, color=color, alpha=0.7, label=smiles)

                            if enable_peak_finding:
                                # Detect peaks with simplified parameters
                                detected_peaks, detected_properties = detect_peaks(
                                    normalized_spectra,
                                    sensitivity=peak_sensitivity,
                                    max_peaks=max_peaks
                                )

                                # Handle overlapping peaks by creating a set of unique wavelengths
                                unique_peak_wavelengths = {}
                                for peak in detected_peaks:
                                    peak_wavelength = x_axis_binned[peak]
                                    # Round wavelength to one decimal to group similar peaks
                                    rounded_wavelength = round(peak_wavelength, 1)
                                    if rounded_wavelength not in unique_peak_wavelengths:
                                        unique_peak_wavelengths[rounded_wavelength] = peak

                                # Label the detected peaks with yellow background and black text
                                for rounded_wavelength, peak in unique_peak_wavelengths.items():
                                    peak_wavelength = x_axis_binned[peak]
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
                        ax_spec.set_xlim([x_axis_binned.min(), x_axis_binned.max()])

                        major_ticks = [3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 20]
                        ax_spec.set_xticks(major_ticks)
                        ax_spec.set_xticklabels([str(tick) for tick in major_ticks])

                        ax_spec.tick_params(direction="in",
                            labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                            bottom=True, top=True, left=True, right=True)

                        ax_spec.set_xlabel("Wavelength ($\mu$m)", fontsize=22)
                        ax_spec.set_ylabel("Absorbance (Normalized to 1)", fontsize=22)

                        if selected_smiles:
                            # Remove duplicate labels in legend
                            handles, labels = ax_spec.get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            ax_spec.legend(by_label.values(), by_label.keys())

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
