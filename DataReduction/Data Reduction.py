import csv
import shutil
import numpy as np
from astropy.io import ascii


def redshift_correction(spectra_dir_path, spectra_names, redshift, destination_dir):

    """
    :param spectra_dir_path: Path to the directory in which are stored the spectra
    :param spectra_names: List of names of spectra. Must be a list()
    :param redshift: List of redshift for every spectra. Must be a list() and its length must be equal to spectra_names
                     length
    :param destination_dir: Path to the directory in which you want to store the corrected spectra
    :return:
    """

    # check if the two input arrays have the same length
    if len(spectra_names) != len(redshift):
        return 'Error: spectra_names and redshift arrays must have same length'

    # redshift correction
    for i in range(len(spectra_names)):
        wavelength_corrected = list()
        # handling exceptions
        try:
            spectrum = ascii.read(spectra_dir_path + spectra_names[i])
            wavelength_column = spectrum['Wavelength']
            flux_column = spectrum['Flux']
            # division of every wavelength of the current spectrum for (1+z)
            for j in range(len(wavelength_column)):
                wavelength_corrected.append((wavelength_column[j]) / (1 + redshift[i]))
            # endfor

            corrected_spectrum = zip(np.around(wavelength_corrected, decimals=3), flux_column)
            corr_spectrum_name = destination_dir + spectra_names[i]
            # creation of a new ascii file  containing the corrected wave and flux in the specified directory
            with open(corr_spectrum_name, 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(corrected_spectrum)
            f.close()
        # endfor

        except FileNotFoundError:
            print('Cannot find file named', spectra_names[i], 'in the directory:', spectra_dir_path)


def continuum_normalization(spectra_dir_path, spectra_names, normal_wave, destination_dir):

    """
    :param spectra_dir_path: Path to the directory in which are stored the spectra
    :param spectra_names: List of names of spectra. Must be a list()
    :param normal_wave: Wavelength corresponding to flux value used for normalization
    :param destination_dir: Path to the directory in which you want to store the corrected spectra
    :return:
    """

    for i in range(len(spectra_names)):
        try:
            spectrum = np.genfromtxt(spectra_dir_path + spectra_names[i])
            # converts array to list and keeps only 3 decimals
            wavelength_column = list(np.around(spectrum[:, 0], decimals=3))
            flux_column = spectrum[:, 1]

            # function to find the closest wavelength value to 5100 and its index
            norm_finding = min(enumerate(wavelength_column), key=lambda x: abs(x[1] - normal_wave))
            # print('Wavelength found for normalization:', norm_finding[1])
            norm_flux = flux_column[norm_finding[0]]  # flux for normalization

            # normalization step
            for j in range(len(flux_column)):
                flux_column[j] = flux_column[j] / norm_flux
            # endfor

            normalized_spectrum = zip(np.around(spectrum[:, 0], decimals=3), np.around(flux_column, decimals=3))
            norm_spectrum_name = destination_dir + spectra_names[i]
            # creation of a new ascii file containing the wave and normalized flux in the specified directory
            with open(norm_spectrum_name, 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(normalized_spectrum)
            f.close()
            if i % 200 == 0:
                print(i)
        # endfor

        except OSError:
            print('Cannot find file named', spectra_names[i], 'in the directory:', spectra_dir_path)


def min_max_wl_list(data_list, file_name, wave_column=''):

    """
    :Description: Creates a .txt file with two columns, one containing the minimum wavelength values for every spectra,
                  and the other the maximum values
    :param data_list: List of path of spectra
    :param file_name: File name that will contain the minimum an maximum wavelengths. Must be .txt
    :param wave_column: Name of the wavelength column in the spectra files. Default to no name
    :return:
    """

    max_wavel = list()
    min_wavel = list()
    if wave_column == '':
        for i in range(len(data_list)):
            spectrum = ascii.read(data_list[i], guess=True)
            max_wavel.append(np.max(spectrum['col1']))
            min_wavel.append(np.min(spectrum['col1']))
            if i % 100 == 0:
                print(i)
        # endfor
    else:
        for i in range(len(data_list)):
            spectrum = ascii.read(data_list[i], guess=True)
            col1 = wave_column
            max_wavel.append(np.max(spectrum[col1]))
            min_wavel.append(np.min(spectrum[col1]))
            if i % 100 == 0:
                print(i)
        # endfor

    max_and_min_wave = zip(max_wavel, min_wavel)

    with open(file_name, 'w') as f:
        writer_rows = csv.writer(f, delimiter='\t')
        writer_rows.writerows(max_and_min_wave)
    f.close()


def min_max_finding(input_list):

    """
    :Description: Finds the minimum and maximum wavelengths to interpolate
    :param input_list: Input list of maximum and minimum wavelengths. Must be .txt
    :return: Minimum value of the maximum wavelengths and maximum value of minimum wavelengths
    """

    max_w = np.loadtxt(fname=input_list, skiprows=1, usecols=0)
    min_w = np.loadtxt(fname=input_list, skiprows=1, usecols=1)
    min_max_wavel = np.min(max_w)
    max_min_wavel = np.max(min_w)
    print("Minimum value of the maximum wavelengths:", min_max_wavel)
    print("Maximum value of the minimum wavelengths:", max_min_wavel, "\n")

    return min_max_wavel, max_min_wavel


def data_interpolation(data_list, min_interp, max_interp, num_interp_points, wave_col=None, flux_col=None):

    """
    :Description: function to interpolate spectra
    :param data_list: List of path of spectra to interpolate
    :param min_interp: Minimum wavelength for the interpolation
    :param max_interp: Maximum wavelength for the interpolation
    :param num_interp_points: Number of points of the interpolation
    :param wave_col: Name of wavelength column in the spectra files. Default to None
    :param flux_col: Name of flux column in the spectra files. Default to None
    :return: interpolation wavelengths (length equal to num_interp_points) and interpolated spectra
    """

    interp_spectra = np.zeros([len(data_list), num_interp_points])

    if wave_col is None and flux_col is None:
        for i in range(len(data_list)):
            # reads wavelength and fluxes for every spectra
            spectrum_test = ascii.read(data_list[i], guess=True)
            wave = np.asarray(spectrum_test['col1'])
            flux = np.asarray(spectrum_test['col2'])

            # interpolation points from min_interp to max_interp
            x_points = np.linspace(min_interp, max_interp, num_interp_points)
            y_points = np.interp(x_points, wave, flux)

            # interpolated spectra
            interp_spectra[i] = np.around(y_points, decimals=3)
            x_points_trunc = np.around(x_points, decimals=3)
            # controls if the function is running every 200 spectra processed
            if i % 200 == 0:
                print(i)
        # endfor
    else:
        for i in range(len(data_list)):
            # reads wavelength and fluxes for every spectra
            spectrum_test = ascii.read(data_list[i], guess=True)
            wave = np.asarray(spectrum_test[wave_col])
            flux = np.asarray(spectrum_test[flux_col])

            # interpolation points from min_interp to max_interp
            x_points = np.linspace(min_interp, max_interp, num_interp_points)
            y_points = np.interp(x_points, wave, flux)

            # interpolated spectra
            interp_spectra[i] = np.around(y_points, decimals=3)
            x_points_trunc = np.around(x_points, decimals=3)
            # controls if the function is running every 200 spectra processed
            if i % 200 == 0:
                print(i)
        # endfor
    # endif
    return x_points_trunc, interp_spectra


# removes the spectra which minimum or maximum wavelength differs too much from the others in the sample.
# Questa funzione è ancora da migliorare. Si dovrebbe aggiungere un try except per tenere conto degli errori.

def data_removing_int(input_list, destination_path, data_list):

    """
    :Description: loads the values contained in the input file and finds the maximum of minimum wavelengths and the
                  minimum of the maximum. After that asks if the spectra containing the minimum, or the one containing
                  the maximum or both, are to be removed from the list. At the end of the process, the array containing
                  the type 1 or type 2 needs to be updated due to the modifications done by this function.
    :param input_list: Input list of minimum and maximum wavelengths for every spectra
    :param destination_path: Path of the folder in which to store the removed spectra
    :param data_list: List of spectra
    :return:
    """

    initial_list = np.genfromtxt(input_list)
    wave_initial = initial_list[:, 0]

    for i in range(len(wave_initial)):
        max_w = np.loadtxt(fname=input_list, usecols=0)
        min_w = np.loadtxt(fname=input_list, usecols=1)
        min_max_wavel = np.min(max_w)
        max_min_wavel = np.max(min_w)
        print("Minimum value of the maximum wavelenghts:", min_max_wavel)
        print("Maximum value of the minimum wavelenghts:", max_min_wavel, "\n")

        maxmin_position = np.where(min_w == max_min_wavel)
        minmax_position = np.where(max_w == min_max_wavel)
        minmax_number = int(minmax_position[0])
        maxmin_number = int(maxmin_position[0])
        print("Row number corresponding to the wavelength", min_max_wavel, ":", minmax_number)
        print("Row number corresponding to the wavelength", max_min_wavel, ":", maxmin_number, "\n")

        # print("Name of the spectra containing the minimum of the maximum:", qso_names_dr14[maxmin_number])
        # print("Name of the spectra containing the maximum of the minimum:", qso_names_dr14[minmax_number], "\n")

        min_removal = input("Do you want to remove the spectra containing the minimum of the maximum?(y/n): ")
        if min_removal == "y":
            with open(input_list, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                # next(reader)                                             #skip header
                minmax_rows = list(reader)
                minmax_rows = np.array(minmax_rows).astype(float)
            f.close()
            minmax_rows_del = np.delete(minmax_rows, minmax_number, axis=0)
            with open(input_list, 'w') as g:
                writer = csv.writer(g, delimiter='\t')
                writer.writerows(minmax_rows_del)
            g.close()
            print("Spectrum removed from list!")
            destination1 = destination_path
            source1 = data_list[minmax_number]
            dest1 = shutil.move(source1, destination1)
            print("Spectrum moved to:", dest1)
        elif min_removal == "n":
            print("Minimum not removed!")
        else:
            pass
        # endif

        max_removal = input("Do you want to remove the spectra containing the maximum of the minimum?(y/n): ")
        if max_removal == "y":
            with open(input_list, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                # next(reader) #skip header
                minmax_rows = list(reader)
                minmax_rows = np.array(minmax_rows).astype(float)
            f.close()
            minmax_rows_del = np.delete(minmax_rows, maxmin_number, axis=0)
            with open(input_list, 'w') as g:
                writer = csv.writer(g, delimiter='\t')
                writer.writerows(minmax_rows_del)
            g.close()
            print("Spectrum removed from list!")
            destination2 = destination_path
            source2 = data_list[maxmin_number]
            dest2 = shutil.move(source2, destination2)
            print("Spectrum moved to:", dest2)
        elif max_removal == "n":
            print("Maximum not removed!")
        else:
            pass
        # endif
        early_ending = input('Do you want to stop the procedure early? (y/n)')
        if early_ending == 'y':
            break
        else:
            pass
        # endif
    # endfor


