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

if 'functional_groups' not in st.session_state:
    st.session_state['functional_groups'] = []

if 'functional_groups_dict' not in st.session_state:
    # Updated SMARTS patterns to use implicit hydrogens
    st.session_state['functional_groups_dict'] = {
        "C-H": "[CH]",
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
        "C=C": "[C]=[C]",
        "C#C": "[C]#[C]"
    }

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

# Initialize session state for functional groups based on user
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
        <li><b>Background Functional Groups:</b> Select or add functional groups to designate background molecules.</li>
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
        <li><b>Filtering:</b> Select functional groups to filter and designate background molecules.</li>
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
    - peaks: Indices of detected peaks.
    - properties: Properties of the detected peaks.
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

    return normalized_spectra, x_axis, highest_peak_idx, highest_peak_intensity

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
        "P-H": "[PH]"
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
            tabs = st.tabs(["Background Gas Settings", "Binning", "Peak Detection", "Sonogram"])

            # ---------------------------
            # Background Gas Settings Tab
            # ---------------------------
            with tabs[0]:
                st.subheader("Background Gas Settings")

                # **Functional Groups Filtering**
                st.markdown("**Functional Groups Filtering**")
                functional_groups = st.session_state['functional_groups_dict']

                # User selects multiple functional groups
                selected_fg = st.multiselect(
                    "Select Functional Groups for Background Molecules:",
                    options=list(functional_groups.keys()),
                    default=[],
                    help="Choose one or more functional groups to designate background molecules based on their structural features."
                )

                # **Background Molecule Filtering**
                background_smiles = []
                if selected_fg:
                    # Initialize background_smiles as all SMILES to perform intersection
                    background_smiles = set(data['SMILES'].unique())
                    for fg_label in selected_fg:
                        fg_smarts = functional_groups.get(fg_label)
                        if fg_smarts:
                            bg_smiles = set(filter_molecules_by_functional_group(data['SMILES'].unique(), fg_smarts))
                            background_smiles = background_smiles.intersection(bg_smiles)
                    background_smiles = list(background_smiles)
                    st.write(f"Selected {len(background_smiles)} molecules as background based on selected functional groups.")
                else:
                    background_smiles = []  # Will handle default later

                # **Add Custom Functional Group**
                st.markdown("**Add Custom Functional Group**")
                add_custom_fg = st.checkbox(
                    "Add Custom Functional Group",
                    help="Enable to add a custom functional group by providing a label and its SMARTS pattern."
                )
                if add_custom_fg:
                    col_fg1, col_fg2 = st.columns([1, 2])
                    with col_fg1:
                        custom_fg_label = st.text_input(
                            "Custom Functional Group Label:",
                            key='custom_fg_label',
                            help="Provide a unique label for the custom functional group."
                        )
                    with col_fg2:
                        custom_fg_smarts = st.text_input(
                            "Custom Functional Group SMARTS:",
                            key='custom_fg_smarts',
                            help="Enter a valid SMARTS pattern that defines the custom functional group."
                        )
                    if st.button("Add Custom Functional Group"):
                        if custom_fg_label and custom_fg_smarts:
                            # Validate the SMARTS pattern before adding
                            fg_test = Chem.MolFromSmarts(custom_fg_smarts)
                            if fg_test:
                                st.session_state['functional_groups_dict'][custom_fg_label] = custom_fg_smarts
                                st.success(f"Added custom functional group: {custom_fg_label}")
                            else:
                                st.error("Invalid SMARTS pattern provided. Please enter a valid SMARTS.")
                        else:
                            st.error("Please provide both label and SMARTS pattern for the custom functional group.")

                # **Background Molecule Opacity Control**
                st.markdown("**Background Molecule Opacity**")
                background_opacity = st.slider(
                    'Set Background Molecule Opacity:',
                    min_value=0.0,
                    max_value=1.0,
                    value=0.01,
                    step=0.01,
                    help="Adjust the transparency of background molecules. Lower values make them more transparent."
                )

            # ---------------------------
            # Binning Tab
            # ---------------------------
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

            # ---------------------------
            # Peak Detection Tab
            # ---------------------------
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

            # ---------------------------
            # Sonogram Tab
            # ---------------------------
            with tabs[3]:
                st.subheader("Sonogram")

                # Plot Sonogram Checkbox
                st.session_state['plot_sonogram'] = st.checkbox(
                    'Plot Sonogram for All Molecules',
                    value=False,
                    help="Check to generate a sonogram plot visualizing spectral differences across all molecules."
                )

        # Foreground Molecules Selection (Clean Interface)
        selected_smiles = st.multiselect('Select Foreground Molecules:', data['SMILES'].unique())

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
                                ax_sono.imshow(np.array(intensity_data)[res_order], aspect='auto', extent=[4000, 500, len(ordered_dist_mat), 0], cmap='viridis')
                                ax_sono.set_xlabel("Wavenumber (cm⁻¹)")
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
                        color_options = ['r', 'g', 'b', 'c', 'm', 'y']
                        random.shuffle(color_options)
                        target_spectra = {}

                        # Automatically select all background molecules if none specified
                        if not background_smiles:
                            background_smiles = data['SMILES'].unique()

                        # First plot background molecules
                        for smiles in background_smiles:
                            spectra_row = data[data['SMILES'] == smiles]
                            if spectra_row.empty:
                                continue
                            spectra = spectra_row.iloc[0]['Raw_Spectra_Intensity']
                            # Apply binning and normalization
                            normalized_spectra, x_axis, _, _ = bin_and_normalize_spectra(
                                spectra, 
                                bin_size=bin_size, 
                                bin_type=bin_type.lower(),  # Now 'wavelength' or 'none'
                                q_branch_threshold=0.3,  # Fixed threshold for automatic normalization
                                max_peak_limit=0.7,
                                debug=False  # Disable debug mode for regular plotting
                            )
                            # Plot background molecule
                            ax_spec.fill_between(x_axis, 0, normalized_spectra, color="k", alpha=background_opacity)

                        # Then plot foreground molecules to ensure they are on top
                        for idx, smiles in enumerate(selected_smiles):
                            spectra_row = data[data['SMILES'] == smiles]
                            if spectra_row.empty:
                                continue
                            spectra = spectra_row.iloc[0]['Raw_Spectra_Intensity']
                            # Apply binning and normalization
                            normalized_spectra, x_axis, _, _ = bin_and_normalize_spectra(
                                spectra, 
                                bin_size=bin_size, 
                                bin_type=bin_type.lower(),  # Now 'wavelength' or 'none'
                                q_branch_threshold=0.3,  # Fixed threshold for automatic normalization
                                max_peak_limit=0.7,
                                debug=False  # Disable debug mode for regular plotting
                            )
                            target_spectra[smiles] = normalized_spectra

                            # Plot foreground molecule
                            color = color_options[idx % len(color_options)]
                            ax_spec.fill_between(x_axis, 0, normalized_spectra, color=color, alpha=0.7, label=smiles)

                            if enable_peak_finding:
                                # Detect peaks with simplified parameters
                                detected_peaks, detected_properties = detect_peaks(
                                    normalized_spectra,
                                    sensitivity=peak_sensitivity,
                                    max_peaks=max_peaks
                                )

                                # Label the detected peaks
                                for peak in detected_peaks:
                                    peak_wavelength = x_axis[peak]
                                    peak_intensity = normalized_spectra[peak]
                                    ax_spec.plot(peak_wavelength, peak_intensity, "x", color=color)
                                    ax_spec.text(
                                        peak_wavelength,
                                        peak_intensity + 0.02,
                                        f'{peak_wavelength:.1f} µm',
                                        fontsize=9,
                                        ha='center',
                                        color=color
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
