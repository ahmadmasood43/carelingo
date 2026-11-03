"""Controlled German vocabulary for comparing facts across languages.

The extractor is asked to use these canonical terms; aliases absorb the variants it
still produces ("Desorientiertheit", "Prellung") and the wording of gold facts. A
term that matches nothing is kept as written and only matches itself.

This list stands in for the catalogue a documentation product would maintain. It was
written by the author of the evaluation set, so its coverage of test concepts is
optimistic.
"""

import re

FINDINGS = {
    "Fieber": ["febril", "erhöhte Temperatur", "Hyperthermie"],
    "Schmerzen": ["Schmerz", "Schmerzangabe", "Schmerzäußerung", "Druckschmerz"],
    "Kopfschmerzen": ["Kopfschmerz", "Zephalgie"],
    "Bauchschmerzen": ["Bauchschmerz", "abdominelle Schmerzen"],
    "Rückenschmerzen": ["Rückenschmerz"],
    "Übelkeit": ["Nausea"],
    "Erbrechen": ["erbrochen", "Emesis"],
    "Durchfall": ["Diarrhö", "Diarrhoe", "flüssiger Stuhl", "wässriger Stuhl"],
    "Stuhlgang": ["Stuhl", "Defäkation", "Darmentleerung"],
    "Atemnot": ["Dyspnoe", "Luftnot", "Kurzatmigkeit"],
    "Husten": [],
    "Zyanose": ["Blaufärbung"],
    "Schwindel": ["Vertigo"],
    "Sturz": ["gestürzt"],
    "Kopfverletzung": ["Kopftrauma", "Verletzung am Kopf"],
    "Hämatom": ["Bluterguss", "Prellung", "Prellmarke", "Kontusion"],
    "Rötung": ["Hautrötung", "gerötet", "Erythem"],
    "Ödem": ["Ödeme", "Schwellung", "Schwellungen", "Unterschenkelödeme"],
    "Dekubitus": ["Druckgeschwür", "Druckulkus"],
    "Hautriss": ["Hauteinriss", "Skin Tear"],
    "Hautdefekt": ["offene Wunde", "offene Hautstelle", "Hautläsion", "Wunde", "offene Haut"],
    "Blutung": ["Nachblutung"],
    "Nasenbluten": ["Epistaxis"],
    "Infektionszeichen": ["Infektzeichen", "Entzündungszeichen"],
    "Augensekret": ["Sekret", "Ausfluss"],
    "Verwirrtheit": ["verwirrt", "Desorientiertheit", "Desorientierung", "desorientiert"],
    "Unruhe": ["unruhig", "Agitation"],
    "Angst": ["ängstlich", "Ängstlichkeit"],
    "Beruhigung": ["ruhiger", "beruhigt"],
    "Schlaf": ["geschlafen", "Nachtruhe", "Einschlafen"],
    "Schwitzen": ["schwitzig", "Hyperhidrose", "Schweißausbruch"],
    "Schüttelfrost": [],
    "Bewusstseinsstörung": ["Bewusstlosigkeit", "bewusstlos", "nicht ansprechbar", "Somnolenz"],
    "Schwäche": ["Kraftminderung", "Parese", "Lähmung"],
    "verwaschene Sprache": ["Dysarthrie", "Sprachstörung"],
    "Schluckbeschwerden": ["Dysphagie", "Schluckstörung"],
    "geringe Nahrungsaufnahme": ["Appetitlosigkeit", "Inappetenz", "isst wenig"],
    "trüber Urin": ["Urin trüb"],
    "übelriechender Urin": ["Uringeruch", "Urin übelriechend"],
    "weicher Bauch": ["Bauch weich", "Abdomen weich"],
    "Bewegungseinschränkung": ["eingeschränkte Beweglichkeit"],
    "nächtliches Aufstehen": ["Aufstehen"],
    "Beschwerden": ["Auffälligkeiten", "Besonderheiten"],
    "Müdigkeit": ["Fatigue"],
    "Juckreiz": ["Pruritus"],
    "Brustschmerzen": ["Thoraxschmerz", "Thoraxschmerzen"],
    "Krampfanfall": ["Krampf", "Anfall"],
}

# Terms that assert the opposite of a canonical finding: "Obstipation" present is
# "Stuhlgang" absent, "Haut intakt" is "Hautdefekt" absent.
INVERSE_FINDINGS = {
    "Obstipation": "Stuhlgang",
    "Verstopfung": "Stuhlgang",
    "afebril": "Fieber",
    "fieberfrei": "Fieber",
    "Haut intakt": "Hautdefekt",
    "Hautintaktheit": "Hautdefekt",
    "intakte Haut": "Hautdefekt",
    "ansprechbar": "Bewusstseinsstörung",
    "schmerzfrei": "Schmerzen",
}

CARE_ACTIONS = {
    "Duschen": ["Dusche"],
    "Körperpflege": ["Grundpflege", "Ganzkörperwäsche"],
    "Gesichtswäsche": [],
    "Mundpflege": [],
    "Verbandswechsel": ["Verbandwechsel", "Verband gewechselt", "Wundauflage gewechselt"],
    "Wundversorgung": ["Wunde versorgt"],
    "Lagerung": ["Umlagerung", "Lagerungswechsel", "Positionswechsel", "Lagerungstherapie"],
    "Mobilisation": [],
    "Fersenschutz": [],
    "Hüftprotektor": [],
    "Inkontinenzversorgung": ["Inkontinenzvorlage", "Windelwechsel", "Vorlagenwechsel"],
    "Getränke andicken": ["Getränke angedickt", "Andicken"],
    "Kühlung": ["gekühlt"],
    "Hausarzt informieren": ["Hausarzt informiert", "Arzt informiert", "Hausarzt verständigt"],
    "Hausarztvisite": ["Arztbesuch", "Visite"],
    "Notruf": ["Notarzt", "Rettungsdienst"],
    "Urinprobe": [],
    "Frühstück": [],
    "Mittagessen": [],
    "Abendessen": [],
    "Angehörige informieren": ["Angehörige informiert"],
    "Toilettengang": [],
    "Kompressionsstrümpfe": [],
}

SITES = {
    "Knie": [],
    "Hüfte": ["Trochanter"],
    "Ferse": ["Fersenbereich"],
    "Sakralbereich": ["Kreuzbein", "sakral", "Sakralregion", "Os sacrum", "Steißbein"],
    "Sitzbein": [],
    "Lendenwirbelsäule": ["LWS", "Lendenwirbelsäulenregion"],
    "Unterschenkel": [],
    "Oberarm": [],
    "Unterarm": [],
    "Arm": [],
    "Auge": ["Augen"],
    "Rücken": [],
    "Kopf": [],
    "Bauch": ["Abdomen"],
    "Haut": [],
    "Fuß": [],
    "Hand": [],
}

# A finding on the upper arm is still a finding on the arm.
SITE_PARENTS = {"Oberarm": "Arm", "Unterarm": "Arm"}

SCALES = {
    "NRS": ["VAS", "Schmerzskala", "Schmerzniveau", "Schmerzintensität"],
    "Braden": ["Braden-Skala"],
    "Dekubitus-Kategorie": ["Kategorie", "Stadium", "Grad"],
}

SUBSTANCES = {
    "Trinkmenge": [
        "getrunken",
        "Flüssigkeit",
        "Flüssigkeitszufuhr",
        "Flüssigkeitsaufnahme",
        "Flüssigkeitsbilanz",
        "Getränke",
    ],
    "Orangensaft": [],
    "Tee": [],
    "Trinknahrung": [],
}

UNITS = {
    "/min": ["pro Minute", "bpm", "Schläge/min", "1/min", "min"],
    "°C": ["Grad", "C"],
    "mmHg": [],
    "mg/dl": ["mg/dL"],
    "%": ["Prozent"],
    "mg": ["Milligramm"],
    "ml": ["mL", "Milliliter"],
    "IE": ["E", "Einheiten", "units"],
    # Polish and English unit words occasionally survive extraction.
    "Hub": [
        "Hübe",
        "Inhalation",
        "Inhalationen",
        "Sprühstoß",
        "Sprühstöße",
        "Wdech",
        "Wdechy",
        "Puff",
    ],
    "Tropfen": [],
    "Beutel": ["Btl."],
    "kg": ["Kilogramm"],
    "cm": ["Zentimeter"],
    "µg/h": ["mcg/h", "ug/h", "Mikrogramm/h"],
    "l/min": ["L/min", "Liter/min"],
}


ROUTES = {
    "s.c.": ["subkutan", "subcutaneous", "podskórnie", "sc"],
    "i.v.": ["intravenös", "intravenous", "dożylnie", "iv"],
    "i.m.": ["intramuskulär", "intramuscular", "domięśniowo", "im"],
    "p.o.": ["oral", "per os", "peroral", "doustnie", "po"],
    "transdermal": ["Pflaster"],
    "inhalativ": ["p.i.", "inhalation", "per inhalationem", "wziewnie"],
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("®", "").replace("-", " ")).strip().lower()


def _index(table: dict[str, list[str]]) -> list[tuple[str, str]]:
    # Longest alias first, so "Rückenschmerzen" wins over "Schmerzen".
    pairs = [
        (_normalize(alias), term) for term, aliases in table.items() for alias in [term, *aliases]
    ]
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


_INDEXES = {
    name: _index(table)
    for name, table in [
        ("finding", FINDINGS),
        ("action", CARE_ACTIONS),
        ("site", SITES),
        ("scale", SCALES),
        ("substance", SUBSTANCES),
        ("unit", UNITS),
        ("route", ROUTES),
    ]
}


def canonical(category: str, text: str | None) -> str | None:
    """Map a term to its canonical form; unknown terms are returned normalized."""
    if text is None:
        return None
    normalized = _normalize(text)
    for alias, term in _INDEXES[category]:
        # Exact match, or the alias opens the phrase ("Lagerung alle 2 Stunden").
        if normalized == alias or normalized.startswith(alias + " "):
            return term
    return normalized


_INVERSE_INDEX = {_normalize(term): target for term, target in INVERSE_FINDINGS.items()}
_FLIPPED = {"present": "absent", "absent": "present", "possible": "possible"}


def canonical_finding(concept: str, polarity: str) -> tuple[str, str]:
    """Canonical concept and polarity, resolving terms that state the opposite."""
    target = _INVERSE_INDEX.get(_normalize(concept))
    if target is not None:
        return target, _FLIPPED[polarity]
    return canonical("finding", concept), polarity


def finding_mentioned(concept: str, text: str) -> bool:
    """Whether a German text names the finding, its synonyms, or its inverse terms."""
    term, _ = canonical_finding(concept, "present")
    forms = {_normalize(term)}
    forms.update(_normalize(alias) for alias in FINDINGS.get(term, []))
    forms.update(_normalize(inverse) for inverse, aim in INVERSE_FINDINGS.items() if aim == term)
    normalized = _normalize(text)
    return any(form in normalized for form in forms)


def is_known(category: str, text: str) -> bool:
    return canonical(category, text) in {term for _, term in _INDEXES[category]}


_SIDE_WORDS = re.compile(r"\b(recht|link|beid)\w*\b\.?\s*|\b(re|li)\.\s*")
_SIDES = {"recht": "right", "re": "right", "link": "left", "li": "left", "beid": "both"}


def site_laterality(site: str | None) -> str | None:
    """Side stated inside a site phrase such as "rechtes Knie" or "li. Hüfte"."""
    match = _SIDE_WORDS.search(_normalize(site)) if site else None
    return _SIDES[match.group(1) or match.group(2)] if match else None


def canonical_site(site: str) -> str | None:
    """Canonical body site, or None for locations that are not body sites ("Bad")."""
    term = canonical("site", _SIDE_WORDS.sub("", _normalize(site)))
    return term if term in SITES else None


def sites_compatible(first: str | None, second: str | None) -> bool:
    a = canonical_site(first) if first else None
    b = canonical_site(second) if second else None
    if a is None or b is None or "Haut" in (a, b):
        # "Haut" names no location, so it is compatible with any body site.
        return True
    return a == b or SITE_PARENTS.get(a) == b or SITE_PARENTS.get(b) == a


def normalize_drug(name: str) -> str:
    return _normalize(name).replace(" pflaster", "").replace("pflaster", "")


def drugs_match(first: str, second: str) -> bool:
    """Same drug, allowing extra words on one side ("Insulin NovoRapid")."""
    a, b = set(normalize_drug(first).split()), set(normalize_drug(second).split())
    return a <= b or b <= a
