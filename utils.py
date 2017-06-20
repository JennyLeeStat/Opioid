import pandas as pd
import os
import sys
import urllib
import logging
import zipfile

logging.basicConfig(
    format= '%(levelname)s %(message)s',
    stream=sys.stdout, level=logging.INFO)



opioids = pd.read_csv("dataset/opioids.csv")
overdose = pd.read_csv("dataset/overdoses.csv", thousands = ',')
prescriber = pd.read_csv("dataset/prescriber-info.csv")

look_up = {
    ".": "",
    ">": "",
    "`": "",
    "-": "",
    "(": "",
    ")": "",
    "-C": "",
    "/": "  ",
    "&": "  ",
    ";": "  ",
    ",": "  "
}

# Hate to do hard coding but trying to save as many as data points as I can here,,,
more_look_up = {
    "M D": "MD",
    "M  D": "MD",
    "MD ": "MD",
    "M. D.": "MD",
    "M. D.,": "MD",
    "M,D, ": "MD",
    "D O": "DO",
    "D,O ": "DO",
    "O D": "OD",
    "O D ": "OD",
    "O D  PS": "OD",
    "N P": "NP",
    "P A": "PA",
    "P,A, ": "PA",
    "PA C": "PA",
    "D P M": "DPM",
    "D D S": "DDS",
    "D. D. S.": "DDS",
    "DD  S": "DDS",
    "D M D": "DMD",
    "PH D": "PHD",
    "PHARM D": "PHARMD",
    "PHYSICIAN ASSISTANT": "PA",
    "NURSE PRACTITIONER": "NP",
    "FAMILY NURSE PRACTIT": "NP",
    "MEDICAL DOCTOR": "MD",
    "MD IN TRAINING": "MD",
    "OPTOMETRIST": "OD"
}

specialty_lookup = {
    'Clinic/Center': 'Other',
    'Preferred Provider Organization': 'Other',
    'Unknown Physician Specialty Code': 'Other',
    'Unknown Supplier/Provider': 'Other',
    'Colorectal Surgery (formerly proctology)': 'Colon & Rectal Surgery',
    'Hematology/Oncology': 'Hematology',
    'Medical Genetics, Ph.D. Medical Genetics': 'Medical Genetics',
    'Maxillofacial Surgery': 'Oral & Maxillofacial Surgery',
    'Orthopaedic Surgery': 'Orthopedic Surgery',
    'Interventional Pain Management': 'Pain Management',
    'Plastic Surgery': 'Plastic and Reconstructive Surgery',
    'Psychiatry & Neurology': 'Psychiatry',
    'Psychologist (billing independently)': 'Psychologist',
    'Specialist/Technologist': 'Specialist',
    'Thoracic Surgery (Cardiothoracic Vascular Surgery)': 'Thoracic Surgery',
    'Hospital (Dmercs Only)': 'Other',
    'Rehabilitation Agency': 'Physical Medicine and Rehabilitation',
    'General Practice': 'Family Practice',
    'Family Medicine': 'Family Practice',
    'Surgery': 'General Surgery',
    'Licensed Practical Nurse': 'Nurse Practitioner'

}
def download_and_decompress(url, dest_dir):

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    filename = url.split('/')[ -1 ]
    filepath = os.path.join(dest_dir, filename)
    uncomp_filedir = filename.split('.')[ 0 ]
    uncomp_filepath = os.path.join(dest_dir, uncomp_filedir)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    if not os.path.isfile(filepath):
        filepath, _ = urllib.request.urlretrieve(url, filepath,
                                                 reporthook=_progress)
        statinfo = os.stat(filepath)
        logging.info('Successfully downloaded {}'.format(filename))
        logging.info('{} bytes.'.format(statinfo.st_size))

    if not os.path.isfile(filepath):
        logging.info("Uncompressing {}".format(filename))
        zipfile.ZipFile(filepath, 'r').extractall(dest_dir)
        logging.info(uncomp_filedir + ' successfully uncompressed')
        print()

    logging.info("Data set {}".format(filename))
    logging.info("from url: {}".format(url))
    logging.info("successfully downloaded and uncompressed!")


def plot_us_map(df, state, code, z,
                title='Overdose Deaths Per Capita',
                colorbar_title=None):

    scl = [ [ 0.0, 'rgb(242,240,247)' ],
            [ 0.2, 'rgb(218,218,235)' ],
            [ 0.4, 'rgb(188,189,220)' ],
            [ 0.6, 'rgb(158,154,200)' ],
            [ 0.8, 'rgb(117,107,177)' ],
            [ 1.0, 'rgb(84,39,143)' ] ]

    for col in df.columns:
        df[ col ] = df[ col ].astype(str)

    df[ 'text' ] = df[state] + '<br>' + z + df[z]

    data = [ dict(
        type='choropleth',
        colorscale=scl,
        autocolorscale=True,
        locations=df[code],
        z=df[z].astype(float),
        locationmode='USA-states',
        text=df[ 'text' ],
        marker=dict(
            line=dict(
                color='rgb(255,255,255)',
                width=2
            )),
        colorbar = dict(
        title=colorbar_title)
    ) ]

    layout = dict(
        title=title,
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=False,
            lakecolor='rgb(255, 255, 255)'),
    )

    fig = dict(data=data, layout=layout)
    return fig

def clean_specialties(specialties):
    specialties_clean = []
    for s in specialties:
        if s in specialty_lookup.keys():
            s = specialty_lookup[s]
        specialties_clean.append(s)
    logging("Finished cleaning Specialty feature")
    return pd.Series(specialties_clean)

def clean_credentials(credentials):
    credentials_clean = [ ]
    for c in credentials:
        new = [ ]
        for s in list(str(c)):
            if s in look_up.keys():
                s = look_up[ s ]
            new.append(s)
        c = "".join(new)
        if c in more_look_up.keys():
            c = more_look_up[ c ]
        credentials_clean.append(c)

    return credentials_clean


def create_credential_variables(credentials_clean):
    MD = [ ]
    DO = [ ]
    Dental = [ ]
    DPM = [ ]
    NP = [ ]
    PA = [ ]
    OD = [ ]

    for _, c in enumerate(credentials_clean):
        is_md = int('MD' in c)
        is_do = int('DO' in c)
        is_dental = int('DDS' in c or 'DMD' in c or 'DENTI' in c)
        is_dpm = int('DPM' in c)
        is_np = int('NP' in c or 'PN' in c or 'PRN' in c)
        is_pa = int('PA' in c or 'PHYSICIAN ASSIS' in c)
        is_od = int('OD' in c)
        MD.append(is_md)
        DO.append(is_do)
        Dental.append(is_dental)
        DPM.append(is_dpm)
        NP.append(is_np)
        PA.append(is_pa)
        OD.append(is_od)

    df = pd.DataFrame({
        'MD': MD,
        'DO': DO,
        'Dental': Dental,
        'DPM': DPM,
        'NP': NP,
        'PA': PA,
        'OD': OD
    })

    return df


def clean_creds(credentials = prescriber[ 'Credentials' ]):
    credentials_clean = clean_credentials(credentials)
    credentials_vars = create_credential_variables(credentials_clean)
    credentials_vars[ 'Other' ] = (credentials_vars.sum(axis=1) == 0).astype(int)
    return credentials_vars


opioids_columns = ['FENTANYL',
     'HYDROCODONE.ACETAMINOPHEN',
     'HYDROMORPHONE.HCL',
     'METHADONE.HCL',
     'MORPHINE.SULFATE',
     'OXYCODONE.HCL',
     'TRAMADOL.HCL',
     'ACETAMINOPHEN.CODEINE',
     'OXYCODONE.ACETAMINOPHEN',
     'MORPHINE.SULFATE.ER',
     'OXYCONTIN',
     'OXYBUTYNIN.CHLORIDE',
     'OXYBUTYNIN.CHLORIDE.ER',
     'ACETAMINOPHEN.CODEINE',
     'OXYCODONE.ACETAMINOPHEN',
     'MORPHINE.SULFATE.ER',
     'OXYCONTIN',
     'OXYBUTYNIN.CHLORIDE',
     'OXYBUTYNIN.CHLORIDE.ER']


























