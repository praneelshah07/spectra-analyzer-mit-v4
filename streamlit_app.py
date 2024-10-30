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
from scipy.signal import find_peaks, peak_widths
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
import uuid

# Set page layout to 'wide' for full screen usage
st.set_page_config(page_title="Spectra Visualization App", layout="wide")

# Adding custom CSS to style the banner, title, and description
st.markdown("""
    <style>
    .banner {
        width: 100%;
        background-color: #89CFF0;  /* You can change the color */
        color: white;
        padding: 20px;
        text-align: center;
        font-size: 45px;  /* Increased font size */
        font-weight: bold;
        margin-bottom: 20px;
    }
    .input-controls-header {
        padding: 10px;
        margin-bottom: 20px;
        background-color: #f0f8ff;
        border: 2px solid #89CFF0;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .graph-header {
        padding: 10px;
        margin-bottom: 20px;
        background-color: #f0f8ff;
        border: 2px solid #89CFF0;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    
    .sidebar {
        font-size: 25px;  /* Sidebar font size */
        line-height: 1.2;  /* Improves readability */
        font-weight: bold;
        text-align: center;
    }

    .description {
        font-size: 14px;  /* Sidebar font size */
        line-height: 1.4;  /* Improves readability */
        color: #333333;
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ddd;  /* Subtle border */
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
    }

    </style>
    """, unsafe_allow_html=True)

# Display the banner across the top
st.markdown('<div class="banner">Spectra Visualization Tool</div>', unsafe_allow_html=True)

# User authentication to enable multi-tenancy
st.sidebar.title("User Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

# Simulated authentication (for demo purposes)
if login_button and username and password:
    user_id = str(uuid.uuid4())  # Assign a unique ID for each user session
    st.session_state['user_id'] = user_id
    st.sidebar.success(f"Logged in as {username}")
else:
    if 'user_id' not in st.session_state:
        st.sidebar.error("Please log in to use the app.")
        st.stop()

user_id = st.session_state['user_id']

# Initialize session state for functional groups based on user
functional_groups_key = f'{user_id}_functional_groups'
if functional_groups_key not in st.session_state:
    st.session_state[functional_groups_key] = []

# Move instructions to the sidebar with improved design
st.sidebar.markdown("""
        <div class="sidebar"> Welcome to the Spectra Visualization Tool. 
        <p><b></b></p>
        </div>
        
        <div class="description">  
        
        <p><b>Here is a breakdown of all the functionalities within the app:</b></p>

        <p><b>Getting Started:</b>
        To get started, either use the pre-loaded dataset or upload your own CSV or ZIP file containing molecular spectra data. Simply select the options that best fit your analysis needs, and confirm your selection to view the corresponding plots and download them as needed.</p> 
        
        <p><b>SMARTS Filtering:</b> 
        Filter molecules by their structural properties using a SMARTS pattern. Enter a SMARTS pattern to refine the dataset.</p> 
        
        <p><b>Advanced Filtering:</b> 
        Search for specific functional groups such as O-H, or C-H. Enter the group to refine the dataset.</p> 
        
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

# Preloaded zip
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

# Function to bin and normalize spectra, with enhanced Q-branch normalization
def bin_and_normalize_spectra(spectra, bin_size, bin_type='wavelength', q_branch_threshold=None, max_peak_limit=0.7):
    st.write("Debug: Entered bin_and_normalize_spectra function.")
    st.write(f"Debug: q_branch_threshold passed to function: {q_branch_threshold}")

    wavenumber = np.arange(4000, 500, -1)
    wavelength = 10000 / wavenumber  # Convert wavenumber to wavelength

    # Bin the spectra if specified
    if 'wavelength' in bin_type:
        bins = np.arange(wavelength.min(), wavelength.max(), bin_size)
        digitized = np.digitize(wavelength, bins)
        x_axis = bins
    else:
        st.write("Debug: No valid binning type provided. Returning without binning.")
        return spectra, None  # No binning if the type is not recognized

    # Perform binning by averaging spectra in each bin
    binned_spectra = np.array([np.mean(spectra[digitized == i]) for i in range(1, len(bins))])
    st.write("Debug: Binning completed.")

    # Enhanced Q-branch handling
    if q_branch_threshold is not None:
        st.write(f"Debug: q_branch_threshold is set: {q_branch_threshold}, proceeding with Q-branch handling.")

        # Detect peaks to identify potential Q-branches
        peaks, properties = find_peaks(binned_spectra, height=q_branch_threshold, prominence=0.3, width=2)

        # Debugging statement to check detected peaks
        st.write(f"Debug: Detected Q-branch peaks at indices: {peaks}")
        st.write(f"Debug: Peak properties: {properties}")

        # Create a copy of the binned spectra for modification
        normalized_spectra = binned_spectra.copy()

        # Cap the intensity of very large Q-branch peaks without affecting other peaks
        for peak in peaks:
            st.write(f"Debug: Handling peak at index {peak} with intensity {normalized_spectra[peak]}")
            if normalized_spectra[peak] > max_peak_limit:
                scaling_factor = max_peak_limit / normalized_spectra[peak]
                normalized_spectra[peak] *= scaling_factor
                st.write(f"Debug: Scaling down peak intensity at index {peak} by factor {scaling_factor}")

        # Apply local smoothing around the Q-branch
        for peak in peaks:
            if peak > 1 and peak < len(normalized_spectra) - 2:
                # Apply simple smoothing by averaging the values around the Q-branch
                st.write(f"Debug: Smoothing peak at index {peak}")
                normalized_spectra[peak] = np.mean([normalized_spectra[peak - 1], normalized_spectra[peak], normalized_spectra[peak + 1]])

        # Further smooth the entire spectrum to minimize sharp Q-branch effects
        smoothed_spectra = np.convolve(normalized_spectra, np.ones(5)/5, mode='same')

        # Normalize the spectra to a max value of 1
        max_value = np.max(smoothed_spectra)
        st.write(f"Debug: Maximum value of smoothed spectra: {max_value}")
        if max_value > 0:
            normalized_spectra = smoothed_spectra / max_value
    else:
        # Standard normalization if Q-branch handling is not specified
        st.write("Debug: q_branch_threshold is None, skipping Q-branch handling.")
        max_peak = np.max(binned_spectra)
        if max_peak > 0:
            normalized_spectra = binned_spectra / max_peak
        else:
            normalized_spectra = binned_spectra

    return normalized_spectra, x_axis[:-1]

# Function to filter molecules by functional group using SMARTS
@st.cache_data
def filter_molecules_by_functional_group(smiles_list, functional_group_smarts):
    filtered_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(functional_group_smarts)):
            filtered_smiles.append(smiles)
    return filtered_smiles

# Enhanced Advanced Filtering with Error Handling
@st.cache_data
def advanced_filtering_by_bond(smiles_list, bond_pattern):
    filtered_smiles = []
    
    # Ensure the bond pattern is recognized and valid
    if bond_pattern == "C-H":
        bond_smarts = "[C][H]"  # SMARTS for C-H bond

    elif bond_pattern == "C=C":
        bond_smarts = "[C]=[C]"  # SMARTS for a carbon-carbon double bond

    elif bond_pattern == "C#C":
        bond_smarts = "[C]#[C]"  # SMARTS for a carbon-carbon triple bond
    
    elif bond_pattern == "O-H":
        bond_smarts = "[O][H]"  # SMARTS for a hydroxyl group (O-H)
    
    elif bond_pattern == "N-H":
        bond_smarts = "[N][H]"  # SMARTS for a nitrogen-hydrogen bond (amine)
    
    elif bond_pattern == "C=O":
        bond_smarts = "[C]=[O]"  # SMARTS for a carbonyl group (C=O)
    
    elif bond_pattern == "C-O":
        bond_smarts = "[C][O]"  # SMARTS for a carbon-oxygen single bond (ether, alcohol)
    
    elif bond_pattern == "C#N":
        bond_smarts = "[C]#[N]"  # SMARTS for a nitrile group (C≡N)
    
    elif bond_pattern == "S-H":
        bond_smarts = "[S][H]"  # SMARTS for a sulfur-hydrogen bond (thiol)
    
    elif bond_pattern == "N=N":
        bond_smarts = "[N]=[N]"  # SMARTS for an azo group (N=N)
    
    elif bond_pattern == "C-S":
        bond_smarts = "[C][S]"  # SMARTS for a carbon-sulfur single bond (thioether)
    
    elif bond_pattern == "C=N":
        bond_smarts = "[C]=[N]"  # SMARTS for an imine group (C=N)
    
    elif bond_pattern == "P-H":
        bond_smarts = "[P][H]"  # SMARTS for a phosphorus-hydrogen bond (phosphine)

    else:
        try:
            bond_smarts = bond_pattern  # Use the input directly for other bond patterns like C=C or C#C
            if not Chem.MolFromSmarts(bond_smarts):
                raise ValueError(f"Invalid SMARTS pattern: {bond_smarts}")
        except Exception as e:
            st.error(f"Error with SMARTS pattern: {e}")
            return filtered_smiles  # Return an empty list in case of error
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol:
            mol_with_h = Chem.AddHs(mol)  # Add explicit hydrogens
            
            if mol_with_h and Chem.MolFromSmarts(bond_smarts):
                # Now check for substructure matches
                if mol_with_h.HasSubstructMatch(Chem.MolFromSmarts(bond_smarts)):
                    filtered_smiles.append(smiles)
        else:
            st.warning(f"Could not process SMILES: {smiles}")
    
    return filtered_smiles

# Compute the distance matrix
def compute_serial_matrix(dist_mat, method="ward"):
    if dist_mat.shape[0] < 2:
        raise ValueError("Not enough data for clustering. Ensure at least two molecules are present.")
    
    res_linkage = linkage(dist_mat, method=method)
    res_order = leaves_list(res_linkage)  # This will give the correct order of leaves

    # Reorder distance matrix based on hierarchical clustering leaves
    ordered_dist_mat = dist_mat[res_order, :][:, res_order]
    return ordered_dist_mat, res_order, res_linkage

# Set up two-column layout below the banner
col1, main_col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="input-controls-header">Input Controls</div>', unsafe_allow_html=True)

    # Load preloaded data from ZIP
    data = load_data_from_zip(ZIP_URL)
    if data is not None:
        st.write("Using preloaded data from GitHub zip file.")

    # File uploader for custom datasets
    uploaded_file = st.file_uploader("If you would like to enter another dataset, insert it here", type=["csv", "zip"])

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
                        data = pd.read_csv(f, usecols=["Formula", "IUPAC chemical name", "SMILES", "Molecular Weight", "Boiling Point (oC)", "Raw_Spectra_Intensity"])
                    st.write(f"Extracted and reading: {csv_file}")
                else:
                    st.error("No CSV file found inside the uploaded ZIP.")
        
        elif uploaded_file.name.endswith('.csv'):
            try:
                data = pd.read_csv(uploaded_file, usecols=["Formula", "IUPAC chemical name", "SMILES", "Molecular Weight", "Boiling Point (oC)", "Raw_Spectra_Intensity"])
                st.write("Using uploaded CSV file data.")
            except pd.errors.EmptyDataError:
                st.error("Uploaded CSV is empty.")
            except KeyError:
                st.error("Uploaded file is missing required columns.")

    # Display dataset preview
    if data is not None:
        data['Raw_Spectra_Intensity'] = data['Raw_Spectra_Intensity'].apply(json.loads)
        data['Raw_Spectra_Intensity'] = data['Raw_Spectra_Intensity'].apply(np.array)
        
        columns_to_display = ["Formula", "IUPAC chemical name", "SMILES", "Molecular Weight", "Boiling Point (oC)"]
        st.write(data[columns_to_display])
   
        # Ensure filtered_smiles is always initialized outside the expander
        filtered_smiles = data['SMILES'].unique()  # Initialize with all available SMILES
        
        # UI Rearrangement with expander for advanced features
        with st.expander("Advanced Filtration Metrics"):
            # Step 1: Filter Selection
            use_smarts_filter = st.checkbox('Apply SMARTS Filtering')
            use_advanced_filter = st.checkbox('Apply Advanced Filtering')
        
            # Step 2: Apply SMARTS filtering if enabled
            if use_smarts_filter:
                functional_group_smarts = st.text_input("Enter a SMARTS pattern to filter molecules:", "")
                if functional_group_smarts:
                    try:
                        filtered_smiles = filter_molecules_by_functional_group(data['SMILES'].unique(), functional_group_smarts)
                        st.write(f"Filtered dataset to {len(filtered_smiles)} molecules using SMARTS pattern.")
                    except Exception as e:
                        st.error(f"Invalid SMARTS pattern: {e}")
        
            # Step 2 (Advanced): Apply advanced filtering if selected
            if use_advanced_filter:
                bond_input = st.text_input("Enter a bond type (e.g., C-C, C#C, C-H):", "")
                if bond_input:
                    filtered_smiles = advanced_filtering_by_bond(data['SMILES'].unique(), bond_input)
                    st.write(f"Filtered dataset to {len(filtered_smiles)} molecules with bond pattern '{bond_input}'.")
        
            # Step 4: Bin size input (only for wavelength)
            bin_type = st.selectbox('Select binning type:', ['None', 'Wavelength (in microns)'])
        
            # Removed the restrictive range for bin sizes
            bin_size = st.number_input('Enter bin size (resolution) in microns:', value=0.1)
        
            # Step 5: Enable Peak Finding and Conditional Dropdown for Peak Detection and Labels
            peak_finding_enabled = st.checkbox('Enable Peak Finding and Labeling', value=False)
            
            if peak_finding_enabled:
                # Remove the nested expander, just show this section without nesting
                st.write("Peak Detection and Background Gas Labels")
                num_peaks = st.slider('Number of Prominent Peaks to Detect', min_value=1, max_value=10, value=5)
            
                # Step 7: Functional group input for background gas labeling (in wavelength)
                st.write("Background Gas Functional Group Labels")
            
                # Form to input functional group data based on wavelength
                with st.form(key='functional_group_form'):
                    fg_label = st.text_input("Functional Group Label (e.g., C-C, N=C=O)")
                    fg_wavelength = st.number_input("Wavelength Position (µm)", min_value=3.0, max_value=20.0, value=12.4)  # Wavelength input
                    add_fg = st.form_submit_button("Add Functional Group")
            
                if add_fg:
                    st.session_state[functional_groups_key].append({'Functional Group': fg_label, 'Wavelength': fg_wavelength})
            
                # Display existing functional group labels and allow deletion
                st.write("Current Functional Group Labels:")
                for i, fg in enumerate(st.session_state[functional_groups_key]):
                    label_col1, label_col2, delete_col = st.columns([2, 2, 1])  # Rename the columns here to avoid naming conflicts
                    label_col1.write(f"Functional Group: {fg['Functional Group']}")
                    label_col2.write(f"Wavelength: {fg['Wavelength']} µm")
                    if delete_col.button(f"Delete", key=f"delete_fg_{i}"):
                        st.session_state[functional_groups_key].pop(i)

            # Step 6: Plot Sonogram (Outside of Expander)
            plot_sonogram = st.checkbox('Plot Sonogram for All Molecules', value=False)
        
    # Background gas selection
    background_smiles = st.multiselect('Select Background Molecules:', data['SMILES'].unique())

    # Background molecule opacity control
    background_opacity = st.slider('Background Molecule Opacity (Default: 0.01)', min_value=0.0, max_value=1.0, value=0.01, step=0.01)

    # The molecule selection (outside the expander)
    selected_smiles = st.multiselect('Select Foreground Molecules:', data['SMILES'].unique())

    # Step 8: Confirm button
    confirm_button = st.button('Confirm Selection and Start Plotting')

with main_col2:
    st.markdown('<div class="graph-header">Graph</div>', unsafe_allow_html=True)

    if confirm_button:
    with st.spinner('Generating plots, this may take some time...'):
        if plot_sonogram:
            intensity_data = np.array(data[data['SMILES'].isin(filtered_smiles)]['Raw_Spectra_Intensity'].tolist())
            if len(intensity_data) > 1:
                dist_mat = squareform(pdist(intensity_data))
                ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, "ward")

                fig, ax = plt.subplots(figsize=(12, 12))
                ratio = int(len(intensity_data[0]) / len(intensity_data))
                ax.imshow(np.array(intensity_data)[res_order], aspect=ratio, extent=[4000, 500, len(ordered_dist_mat), 0])
                ax.set_xlabel("Wavenumber")
                ax.set_ylabel("Molecules")

                st.pyplot(fig)
                plt.clf()

                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                st.download_button(label="Download Sonogram as PNG", data=buf, file_name="sonogram.png", mime="image/png")
            else:
                st.error("Not enough data to generate the sonogram. Please ensure there are at least two molecules.")
        else:
            top_peaks = []
            fig, ax = plt.subplots(figsize=(16, 6.5), dpi=100)
            wavenumber = np.arange(4000, 500, -1)
            wavelength = 10000 / wavenumber

            color_options = ['r', 'g', 'b', 'c', 'm', 'y']
            random.shuffle(color_options)

            target_spectra = {}
            for smiles, spectra in data[['SMILES', 'Raw_Spectra_Intensity']].values:
                if smiles in selected_smiles:
                    # Apply binning if selected
                    if bin_type != 'None':
                        st.write(f"Debug: Calling bin_and_normalize_spectra for {smiles} with bin_size={bin_size}")
                        spectra, x_axis = bin_and_normalize_spectra(spectra, bin_size, bin_type.lower(), q_branch_threshold=0.1)
                    else:
                        st.write(f"Debug: Normalizing spectra without binning for {smiles}")
                        spectra = spectra / np.max(spectra)  # Normalize if no binning
                        x_axis = wavelength
                    target_spectra[smiles] = spectra
                elif smiles in background_smiles or (not background_smiles and smiles in filtered_smiles):
                    if bin_type != 'None':
                        st.write(f"Debug: Calling bin_and_normalize_spectra for background molecule {smiles} with bin_size={bin_size}")
                        spectra, x_axis = bin_and_normalize_spectra(spectra, bin_size, bin_type.lower(), q_branch_threshold=0.1)
                    else:
                        spectra = spectra / np.max(spectra)  # Normalize if no binning
                        x_axis = wavelength
                    ax.fill_between(x_axis, 0, spectra, color="k", alpha=background_opacity)
           
            for i, smiles in enumerate(target_spectra):
                spectra = target_spectra[smiles]
                ax.fill_between(x_axis, 0, spectra, color=color_options[i % len(color_options)], alpha=0.5, label=f"{smiles}")

                if peak_finding_enabled:
                    # Detect peaks and retrieve peak properties like prominence
                    peaks, properties = find_peaks(spectra, height=0.05, prominence=0.1)
                    st.write(f"Debug: Detected peaks for {smiles} at indices: {peaks}")
                    st.write(f"Debug: Peak properties: {properties}")
                    
                    # Sort the peaks by their prominence and select the top `num_peaks`
                    if len(peaks) > 0:
                        prominences = properties['prominences']
                        # Zip peaks with their corresponding prominences, then sort by prominence
                        peaks_with_prominences = sorted(zip(peaks, prominences), key=lambda x: x[1], reverse=True)
                        # Extract the top `num_peaks` most prominent peaks
                        top_peaks = [p[0] for p in peaks_with_prominences[:num_peaks]]
                        # Now label the top peaks
                        for peak in top_peaks:
                            peak_wavelength = x_axis[peak]
                            peak_intensity = spectra[peak]
                             # Label the peaks with wavelength
                            ax.text(peak_wavelength, peak_intensity + 0.05, f'{round(peak_wavelength, 1)}', 
                                    fontsize=10, ha='center', color=color_options[i % len(color_options)])
                                                         
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

            # Number of label matches
            ax.set_xticklabels([str(tick) for tick in major_ticks])

            ax.tick_params(direction="in",
                labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=True, left=True, right=True)

            ax.set_xlabel("Wavelength ($μm)", fontsize=22)
            ax.set_ylabel("Absorbance (Normalized to 1)", fontsize=22)

            if selected_smiles:
                ax.legend()

            st.pyplot(fig)
    
            # Download button for the spectra plot
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            st.download_button(label="Download Plot as PNG", data=buf, file_name="spectra_plot.png", mime="image/png")
                                                             
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

                # Number of label matches
                ax.set_xticklabels([str(tick) for tick in major_ticks])

                ax.tick_params(direction="in",
                    labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=True, left=True, right=True)

                ax.set_xlabel("Wavelength ($μm)", fontsize=22)
                ax.set_ylabel("Absorbance (Normalized to 1)", fontsize=22)

                if selected_smiles:
                    ax.legend()

                st.pyplot(fig)
        
                # Download button for the spectra plot
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                st.download_button(label="Download Plot as PNG", data=buf, file_name="spectra_plot.png", mime="image/png")
                
