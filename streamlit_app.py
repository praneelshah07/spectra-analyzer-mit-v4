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

# Function to bin and normalize spectra
def bin_and_normalize_spectra(spectra, bin_size, bin_type='wavelength'):
    wavenumber = np.arange(4000, 500, -1)
    wavelength = 10000 / wavenumber  # Convert wavenumber to wavelength

    if bin_type == 'wavelength':
        bins = np.arange(wavelength.min(), wavelength.max(), bin_size)
        digitized = np.digitize(wavelength, bins)
        x_axis = bins
    elif bin_type == 'wavenumber':
        bins = np.arange(wavenumber.min(), wavenumber.max(), bin_size)
        digitized = np.digitize(wavenumber, bins)
        x_axis = bins
    else:
        return spectra, None  # No binning if the type is not recognized

    # Perform binning by averaging spectra in each bin
    binned_spectra = np.array([np.mean(spectra[digitized == i]) for i in range(1, len(bins))])

    # Normalize the spectra after binning
    normalized_spectra = binned_spectra / np.max(binned_spectra)
    
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

# Compute the distance matrix
def compute_serial_matrix(dist_mat, method="ward"):
    if dist_mat.shape[0] < 2:
        raise ValueError("Not enough data for clustering. Ensure at least two molecules are present.")
    
    res_linkage = linkage(dist_mat, method=method)
    res_order = leaves_list(res_linkage)  # This will give the correct order of leaves

    # Reorder distance matrix based on hierarchical clustering leaves
    ordered_dist_mat = dist_mat[res_order, :][:, res_order]
    return ordered_dist_mat, res_order, res_linkage

# Set up
st.title("Spectra Visualization App")

# Load data
data = load_data_from_zip(ZIP_URL)
if data is not None:
    st.write("Using preloaded data from GitHub zip file.")

# File uploader
uploaded_file = st.file_uploader("If you would like to enter another dataset, insert it here", type=["csv", "zip"])

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

if data is not None:
    data['Raw_Spectra_Intensity'] = data['Raw_Spectra_Intensity'].apply(json.loads)
    data['Raw_Spectra_Intensity'] = data['Raw_Spectra_Intensity'].apply(np.array)
    
    columns_to_display = ["Formula", "IUPAC chemical name", "SMILES", "Molecular Weight", "Boiling Point (oC)"]
    st.write(data[columns_to_display])

    unique_smiles = data['SMILES'].unique()

    # Option to filter molecules using SMARTS patterns
    use_smarts_filter = st.checkbox('Apply SMARTS Filtering', value=False)

    # Initialize the filtered dataset and highlight options
    filtered_smiles = unique_smiles

    if use_smarts_filter:
        functional_group_smarts = st.text_input("Enter a SMARTS pattern to filter molecules:", "")
        if functional_group_smarts:
            try:
                filtered_smiles = filter_molecules_by_functional_group(unique_smiles, functional_group_smarts)
                st.write(f"Filtered dataset to {len(filtered_smiles)} molecules using SMARTS pattern.")
            except Exception as e:
                st.error(f"Invalid SMARTS pattern: {e}")

    # Binning options
    bin_type = st.selectbox('Select binning type:', ['None', 'Wavelength', 'Wavenumber'])
    bin_size = st.number_input('Enter bin size (resolution):', min_value=0.01, max_value=1.0, value=0.1)

    # Multiselect for highlighting molecules (now using the filtered list)
    selected_smiles = st.multiselect('Select molecules by SMILES to highlight:', filtered_smiles)

    # Peak detection settings
    peak_finding_enabled = st.checkbox('Enable Peak Finding and Labeling', value=False)
    num_peaks = st.slider('Number of Peaks to Detect', min_value=1, max_value=20, value=5)

    # Functional group input for background gas labeling
    st.write("Background Gas Functional Group Labels")
    functional_groups = st.experimental_data_editor(
        pd.DataFrame({'Functional Group': [], 'Wavenumber': []}),
        key='functional_groups',
        use_container_width=True
    )

    # Sonogram plotting using all data
    plot_sonogram = st.checkbox('Plot Sonogram for All Molecules', value=False)

    confirm_button = st.button('Confirm Selection and Start Plotting')

    if confirm_button:
        with st.spinner('Generating plots, this may take some time...'):
            if plot_sonogram:
                st.write("Generating sonogram, please wait...")

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
                fig, ax = plt.subplots(figsize=(16, 6.5), dpi=100)
                wavenumber = np.arange(4000, 500, -1)
                wavelength = 10000 / wavenumber

                color_options = ['r', 'g', 'b', 'c', 'm', 'y']
                random.shuffle(color_options)

                target_spectra = {}
                for smiles, spectra in data[data['SMILES'].isin(filtered_smiles)][['SMILES', 'Raw_Spectra_Intensity']].values:
                    if smiles in selected_smiles:
                        # Apply binning if selected
                        if bin_type != 'None':
                            spectra, x_axis = bin_and_normalize_spectra(spectra, bin_size, bin_type.lower())
                        else:
                            spectra = spectra / np.max(spectra)  # Normalize if no binning
                            x_axis = wavelength
                        target_spectra[smiles] = spectra
                    else:
                        if bin_type != 'None':
                            spectra, x_axis = bin_and_normalize_spectra(spectra, bin_size, bin_type.lower())
                        else:
                            spectra = spectra / np.max(spectra)  # Normalize if no binning
                            x_axis = wavelength
                        ax.fill_between(x_axis, 0, spectra, color="k", alpha=0.01)

                for i, smiles in enumerate(target_spectra):
                    spectra = target_spectra[smiles]
                    ax.fill_between(x_axis, 0, spectra, color=color_options[i % len(color_options)], 
                                    alpha=0.5, label=f"{smiles}")

                    if peak_finding_enabled:
                        peaks, _ = find_peaks(spectra, height=0.05)
                        peaks = peaks[:num_peaks]  # Limit the number of peaks based on user selection
                        for peak in peaks:
                            peak_wavelength = x_axis[peak]
                            peak_intensity = spectra[peak]
                            # Label peaks with wavelength
                            ax.text(peak_wavelength, peak_intensity + 0.05, f'{round(peak_wavelength, 1)}', 
                                    fontsize=10, ha='center', color=color_options[i % len(color_options)])

                # Add functional group labels for background gases
                for index, row in functional_groups.iterrows():
                    fg_wavenumber = row['Wavenumber']
                    fg_label = row['Functional Group']
                    ax.axvline(fg_wavenumber, color='grey', linestyle='--')
                    ax.text(fg_wavenumber, 1, fg_label, fontsize=12, color='grey', ha='center')

                # Customize plot
                ax.set_xlim([x_axis.min(), x_axis.max()])

                major_ticks = [3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 20]
                ax.set_xticks(major_ticks)

                # Number of label matches
                ax.set_xticklabels([str(tick) for tick in major_ticks])

                ax.tick_params(direction="in",
                    labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=True, left=True, right=True)

                ax.set_xlabel("Wavelength ($\mu$m)" if bin_type == 'Wavelength' else "Wavenumber (cm⁻¹)", fontsize=22)
                ax.set_ylabel("Absorbance (Normalized to 1)", fontsize=22)

                if selected_smiles:
                    ax.legend()

                st.pyplot(fig)
        
                # Download button for the spectra plot
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                st.download_button(label="Download Plot as PNG", data=buf, file_name="spectra_plot.png", mime="image/png")
