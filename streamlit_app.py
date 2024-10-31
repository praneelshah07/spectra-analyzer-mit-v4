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
        line-height: 1.4;  
        color: #333333;
        background-color: #f0f8ff;
        padding: 10px;
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

# Simplified authentication: Only username required
st.sidebar.title("User Login")
username = st.sidebar.text_input("Username")
login_button = st.sidebar.button("Login")

# Simulated authentication (for demo purposes)
if login_button and username:
    user_id = str(uuid.uuid4())  # Assign a unique ID for each user session
    st.session_state['user_id'] = user_id
    st.sidebar.success(f"Logged in as {username}")
elif 'user_id' not in st.session_state:
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

    <p><b>Here is a breakdown of all the functionalities within the app:</b></p>

    <p><b>Getting Started:</b>
    Use the pre-loaded dataset or upload your own CSV or ZIP file containing molecular spectra data. Select the options that best fit your analysis needs, and confirm your selection to view the corresponding plots and download them as needed.</p> 
    
    <p><b>SMARTS Filtering:</b> 
    Filter molecules by their structural properties using a SMARTS pattern. Enter a SMARTS pattern to refine the dataset.</p> 
    
    <p><b>Advanced Filtering:</b> 
    Search for specific functional groups such as O-H, or C-H. Select a bond pattern to refine the dataset.</p> 
    
    <p><b>Binning Feature:</b>
    Bin a certain amount of data within one datapoint to simplify the plot produced.</p>  
    
    <p><b>Peak Detection:</b>  
    Enable this feature to automatically detect and label prominent peaks in the spectra.</p>  
    
    <p><b>Background Gas Labels:</b>  
    Add functional group labels based on wavelengths for easier identification of background gases in your spectra.</p> 
    
    <p><b>Sonogram Plot:</b> 
    View a detailed sonogram plot for all molecules in your dataset to visualize spectral differences across compounds.</p> 
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

    # Enhanced Q-branch handling
    # Detect peaks to identify potential Q-branches
    peaks, properties = find_peaks(
        binned_spectra, 
        height=q_branch_threshold,      # Threshold for peak height
        prominence=0.3,                  # Prominence to filter out noise
        width=2                          # Expected width of Q-branch
    )

    if debug:
        fig_debug, ax_debug = plt.subplots(figsize=(10, 4))
        ax_debug.plot(x_axis, binned_spectra, label='Binned Spectra' if bin_size else 'Original Spectra')
        ax_debug.plot(x_axis[peaks], binned_spectra[peaks], "x", label='Detected Peaks')
        ax_debug.set_title("Peak Detection for Q-Branch")
        ax_debug.set_xlabel("Wavelength (µm)")
        ax_debug.set_ylabel("Intensity")
        ax_debug.legend()
        st.pyplot(fig_debug)

    # Create a copy of the binned spectra for modification
    normalized_spectra = binned_spectra.copy()

    # Cap the intensity of very large Q-branch peaks without affecting other peaks
    for peak in peaks:
        if normalized_spectra[peak] > max_peak_limit:
            scaling_factor = max_peak_limit / normalized_spectra[peak]
            normalized_spectra[peak] *= scaling_factor

    # Apply local smoothing around the Q-branch
    for peak in peaks:
        if 0 < peak < len(normalized_spectra) - 1:
            # Apply simple smoothing by averaging the values around the Q-branch
            normalized_spectra[peak] = np.mean([
                normalized_spectra[peak - 1], 
                normalized_spectra[peak], 
                normalized_spectra[peak + 1]
            ])

    # Further smooth the entire spectrum to minimize sharp Q-branch effects
    smoothed_spectra = np.convolve(normalized_spectra, np.ones(5)/5, mode='same')

    # Normalize the spectra to a max value of 1
    max_value = np.max(smoothed_spectra)
    if max_value > 0:
        normalized_spectra = smoothed_spectra / max_value
    else:
        normalized_spectra = smoothed_spectra

    return normalized_spectra, x_axis, peaks, properties

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
            if mol and mol.HasSubstructMatch(fg_mol):
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
        "C-H": "[C][H]",
        "C=C": "[C]=[C]",
        "C#C": "[C]#[C]",
        "O-H": "[O][H]",
        "N-H": "[N][H]",
        "C=O": "[C]=[O]",
        "C-O": "[C][O]",
        "C#N": "[C]#[N]",
        "S-H": "[S][H]",
        "N=N": "[N]=[N]",
        "C-S": "[C][S]",
        "C=N": "[C]=[N]",
        "P-H": "[P][H]"
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
        if mol and mol.HasSubstructMatch(bond_mol):
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
            # Convert Raw_Spectra_Intensity from JSON strings to numpy arrays
            data['Raw_Spectra_Intensity'] = data['Raw_Spectra_Intensity'].apply(json.loads)
            data['Raw_Spectra_Intensity'] = data['Raw_Spectra_Intensity'].apply(np.array)
        except Exception as e:
            st.error(f"Error processing Raw_Spectra_Intensity: {e}")
            st.stop()

        columns_to_display = ["Formula", "IUPAC chemical name", "SMILES", "Molecular Weight", "Boiling Point (oC)"]
        st.write(data[columns_to_display])

        # Initialize filtered_smiles with all unique SMILES
        filtered_smiles = data['SMILES'].unique()

        # Advanced Filtration Metrics
        with st.expander("Advanced Filtration Metrics"):
            # Step 1: Filter Selection
            use_smarts_filter = st.checkbox('Apply SMARTS Filtering')
            use_advanced_filter = st.checkbox('Apply Advanced Bond Filtering')

            # Step 2: Apply SMARTS filtering if enabled
            if use_smarts_filter:
                functional_group_smarts = st.text_input("Enter a SMARTS pattern to filter molecules:", "")
                if functional_group_smarts:
                    filtered_smiles_smarts = filter_molecules_by_functional_group(data['SMILES'].unique(), functional_group_smarts)
                    st.write(f"Filtered dataset to {len(filtered_smiles_smarts)} molecules using SMARTS pattern.")
                    filtered_smiles = np.intersect1d(filtered_smiles, filtered_smiles_smarts)

            # Step 3: Apply advanced filtering if selected
            if use_advanced_filter:
                bond_input = st.selectbox("Select a bond type to filter molecules:", 
                                          ["None", "C-H", "C=C", "C#C", "O-H", "N-H", 
                                           "C=O", "C-O", "C#N", "S-H", "N=N", "C-S", "C=N", "P-H"])
                if bond_input and bond_input != "None":
                    filtered_smiles_bond = advanced_filtering_by_bond(data['SMILES'].unique(), bond_input)
                    st.write(f"Filtered dataset to {len(filtered_smiles_bond)} molecules with bond pattern '{bond_input}'.")
                    filtered_smiles = np.intersect1d(filtered_smiles, filtered_smiles_bond)

            # Step 4: Functional Groups for Background Selection
            st.markdown("<hr>", unsafe_allow_html=True)
            st.header("Background Functional Groups")
            # Predefined list of common functional groups
            functional_groups = {
                "C-H": "[C][H]",
                "O-H": "[O][H]",
                "N-H": "[N][H]",
                "C=O": "[C]=[O]",
                "C-O": "[C][O]",
                "C#N": "[C]#[N]",
                "S-H": "[S][H]",
                "N=N": "[N]=[N]",
                "C-S": "[C][S]",
                "C=N": "[C]=[N]",
                "P-H": "[P][H]"
            }
            # User selects multiple functional groups
            selected_fg = st.multiselect(
                "Select Functional Groups for Background Molecules:",
                options=list(functional_groups.keys()),
                default=[]
            )
            # Alternatively, allow user to input SMARTS patterns
            add_custom_fg = st.checkbox("Add Custom Functional Group")
            if add_custom_fg:
                custom_fg_label = st.text_input("Custom Functional Group Label:")
                custom_fg_smarts = st.text_input("Custom Functional Group SMARTS:")
                if st.button("Add Custom Functional Group"):
                    if custom_fg_label and custom_fg_smarts:
                        functional_groups[custom_fg_label] = custom_fg_smarts
                        st.success(f"Added custom functional group: {custom_fg_label}")
                    else:
                        st.error("Please provide both label and SMARTS pattern for the custom functional group.")

            # Step 5: Filter Background Molecules based on selected functional groups
            background_smiles = []
            if selected_fg:
                for fg_label in selected_fg:
                    fg_smarts = functional_groups.get(fg_label)
                    if fg_smarts:
                        bg_smiles = filter_molecules_by_functional_group(data['SMILES'].unique(), fg_smarts)
                        background_smiles.extend(bg_smiles)
                background_smiles = list(set(background_smiles))  # Remove duplicates
                st.write(f"Selected {len(background_smiles)} molecules as background based on functional groups.")
            else:
                background_smiles = []  # Will handle default later

            # Step 6: Background molecule opacity control
            st.markdown("<hr>", unsafe_allow_html=True)
            st.header("Background Molecule Opacity")
            background_opacity = st.slider('Background Molecule Opacity (Default: 0.01)', min_value=0.0, max_value=1.0, value=0.01, step=0.01)

            # Step 7: Bin size input (only for wavelength)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.header("Binning Options")
            bin_type = st.selectbox('Select binning type:', ['None', 'Wavelength (µm)'])

            if bin_type == 'Wavelength (µm)':
                bin_size = st.number_input('Enter bin size (µm):', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
            else:
                bin_size = None

            # Step 8: Peak Detection Parameters
            st.markdown("<hr>", unsafe_allow_html=True)
            st.header("Peak Detection Parameters")
            peak_height = st.slider("Peak Height Threshold", 0.0, 1.0, 0.3, 0.05)
            peak_prominence = st.slider("Peak Prominence", 0.0, 1.0, 0.5, 0.05)
            peak_width = st.slider("Peak Width", 1, 10, 2, 1)

            # Step 9: Enable Peak Finding and Conditional Dropdown for Peak Detection and Labels
            peak_finding_enabled = st.checkbox('Enable Peak Finding and Labeling', value=False)
            
            if peak_finding_enabled:
                st.write("Configure Peak Detection Settings:")
                # Allow users to adjust peak detection parameters
                # These are already provided above
                
                # Step 10: Functional group input for background gas labeling (in wavelength)
                st.markdown("<hr>", unsafe_allow_html=True)
                st.header("Background Gas Functional Group Labels")
            
                # Form to input functional group data based on wavelength
                with st.form(key='functional_group_form'):
                    fg_label = st.text_input("Functional Group Label (e.g., C-C, N=C=O)")
                    fg_wavelength = st.number_input("Wavelength Position (µm)", min_value=3.0, max_value=20.0, value=15.0, step=0.1)
                    add_fg = st.form_submit_button("Add Functional Group")
            
                if add_fg:
                    if fg_label and fg_wavelength:
                        st.session_state[functional_groups_key].append({'Functional Group': fg_label, 'Wavelength': fg_wavelength})
                        st.success(f"Added functional group: {fg_label} at {fg_wavelength} µm")
                    else:
                        st.error("Please provide both label and wavelength for the functional group.")
            
                # Display existing functional group labels and allow deletion
                st.write("Current Functional Group Labels:")
                for i, fg in enumerate(st.session_state[functional_groups_key]):
                    label_col1, label_col2, delete_col = st.columns([2, 2, 1])
                    label_col1.write(f"**Functional Group:** {fg['Functional Group']}")
                    label_col2.write(f"**Wavelength:** {fg['Wavelength']} µm")
                    if delete_col.button(f"Delete", key=f"delete_fg_{i}"):
                        st.session_state[functional_groups_key].pop(i)
                        st.success(f"Deleted functional group: {fg['Functional Group']}")

            # Step 11: Plot Sonogram Option
            st.markdown("<hr>", unsafe_allow_html=True)
            st.header("Sonogram Plot")
            plot_sonogram = st.checkbox('Plot Sonogram for All Molecules', value=False)

    # The molecule selection (foreground) is now without header and line
    if data is not None:
        # Removed the header and horizontal line for cleaner interface
        selected_smiles = st.multiselect('Select Foreground Molecules:', data['SMILES'].unique())

        # Step 12: Confirm button
        confirm_button = st.button('Confirm Selection and Start Plotting')

with main_col2:
    st.markdown('<div class="header">Graph</div>', unsafe_allow_html=True)

    if confirm_button:
        if data is None:
            st.error("No data available to plot.")
        else:
            with st.spinner('Generating plots, this may take some time...'):
                if plot_sonogram:
                    # Sonogram plotting logic
                    intensity_data = np.array(data[data['SMILES'].isin(filtered_smiles)]['Raw_Spectra_Intensity'].tolist())
                    if len(intensity_data) > 1:
                        try:
                            dist_mat = squareform(pdist(intensity_data))
                            ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, "ward")

                            fig, ax = plt.subplots(figsize=(12, 12))
                            ax.imshow(np.array(intensity_data)[res_order], aspect='auto', extent=[4000, 500, len(ordered_dist_mat), 0], cmap='viridis')
                            ax.set_xlabel("Wavenumber (cm⁻¹)")
                            ax.set_ylabel("Molecules")
                            ax.set_title("Sonogram Plot")

                            st.pyplot(fig)
                            plt.close(fig)

                            # Download button for the sonogram
                            buf = io.BytesIO()
                            fig.savefig(buf, format='png')
                            buf.seek(0)
                            st.download_button(label="Download Sonogram as PNG", data=buf, file_name="sonogram.png", mime="image/png")
                        except Exception as e:
                            st.error(f"Error generating sonogram: {e}")
                    else:
                        st.error("Not enough data to generate the sonogram. Please ensure there are at least two molecules.")
                else:
                    # Spectra plotting logic
                    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=100)
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
                        normalized_spectra, x_axis, peaks, properties = bin_and_normalize_spectra(
                            spectra, 
                            bin_size=bin_size, 
                            bin_type=bin_type.lower() if bin_type != 'None' else 'none',
                            q_branch_threshold=peak_height,  # Using peak_height as threshold
                            max_peak_limit=0.7,
                            debug=False  # Disable debug mode for regular plotting
                        )
                        # Plot background molecule
                        ax.fill_between(x_axis, 0, normalized_spectra, color="k", alpha=background_opacity)

                    # Then plot foreground molecules to ensure they are on top
                    for smiles in selected_smiles:
                        spectra_row = data[data['SMILES'] == smiles]
                        if spectra_row.empty:
                            continue
                        spectra = spectra_row.iloc[0]['Raw_Spectra_Intensity']
                        # Apply binning and normalization
                        normalized_spectra, x_axis, peaks, properties = bin_and_normalize_spectra(
                            spectra, 
                            bin_size=bin_size, 
                            bin_type=bin_type.lower() if bin_type != 'None' else 'none',
                            q_branch_threshold=peak_height,  # Using peak_height as threshold
                            max_peak_limit=0.7,
                            debug=False  # Disable debug mode for regular plotting
                        )
                        target_spectra[smiles] = normalized_spectra

                        # Plot foreground molecule
                        ax.fill_between(x_axis, 0, normalized_spectra, color=color_options[len(target_spectra) % len(color_options)], alpha=0.7, label=smiles)

                        if peak_finding_enabled:
                            # Detect peaks with user-defined parameters
                            detected_peaks, detected_properties = find_peaks(
                                normalized_spectra, 
                                height=peak_height, 
                                prominence=peak_prominence, 
                                width=peak_width
                            )

                            # Sort the peaks by their prominence and select the top `num_peaks`
                            if len(detected_peaks) > 0:
                                prominences = detected_properties['prominences']
                                peaks_with_prominences = sorted(zip(detected_peaks, prominences), key=lambda x: x[1], reverse=True)
                                num_peaks = st.sidebar.slider('Number of Prominent Peaks to Detect', min_value=1, max_value=10, value=5)
                                top_peaks = [p[0] for p in peaks_with_prominences[:num_peaks]]

                                # Label the top peaks
                                for peak in top_peaks:
                                    peak_wavelength = x_axis[peak]
                                    peak_intensity = normalized_spectra[peak]
                                    # Label the peaks with wavelength
                                    ax.text(peak_wavelength, peak_intensity + 0.05, f'{peak_wavelength:.1f} µm', 
                                            fontsize=10, ha='center', color=color_options[len(target_spectra) % len(color_options)])

                    # Add functional group labels for background gases based on wavelength
                    for fg in st.session_state[functional_groups_key]:
                        fg_wavelength = fg['Wavelength']
                        fg_label = fg['Functional Group']
                        ax.axvline(fg_wavelength, color='grey', linestyle='--')
                        ax.text(fg_wavelength, 1, fg_label, fontsize=12, color='black', ha='center')

                    # Customize plot
                    ax.set_xlim([x_axis.min(), x_axis.max()])

                    major_ticks = [3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 20]
                    ax.set_xticks(major_ticks)
                    ax.set_xticklabels([str(tick) for tick in major_ticks])

                    ax.tick_params(direction="in",
                        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                        bottom=True, top=True, left=True, right=True)

                    ax.set_xlabel("Wavelength ($\mu$m)", fontsize=22)
                    ax.set_ylabel("Absorbance (Normalized to 1)", fontsize=22)

                    if selected_smiles:
                        ax.legend()

                    st.pyplot(fig)

                    # Download button for the spectra plot
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    st.download_button(label="Download Plot as PNG", data=buf, file_name="spectra_plot.png", mime="image/png")
