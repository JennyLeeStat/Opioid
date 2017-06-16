import pandas as pd

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


























