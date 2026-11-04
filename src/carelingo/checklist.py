"""Show a nurse, in the language they dictated in, what the German note will record.

The point of dictating in one's own language is lost if the nurse then has to judge
a German note they cannot fully read. The checklist renders the facts extracted from
their dictation, and the problems found in the German draft, back in English or
Polish. Rendering uses fixed templates and a translation table for the controlled
vocabulary, never a model, so this step cannot introduce translation errors of its
own. Terms outside the vocabulary are shown in German and marked as such.
"""

import re

from carelingo.facts import (
    CareMeasure,
    Fact,
    Finding,
    Intake,
    Measurement,
    Medication,
    ResidentRef,
    Score,
)
from carelingo.verification import Discrepancy
from carelingo.vocabulary import canonical, canonical_finding, canonical_site

# Canonical German term → (English, Polish).
TERMS = {
    # findings
    "Fieber": ("fever", "gorączka"),
    "Schmerzen": ("pain", "ból"),
    "Kopfschmerzen": ("headache", "ból głowy"),
    "Bauchschmerzen": ("abdominal pain", "ból brzucha"),
    "Rückenschmerzen": ("back pain", "ból pleców"),
    "Übelkeit": ("nausea", "nudności"),
    "Erbrechen": ("vomiting", "wymioty"),
    "Durchfall": ("diarrhoea", "biegunka"),
    "Stuhlgang": ("bowel movement", "wypróżnienie"),
    "Atemnot": ("shortness of breath", "duszność"),
    "Husten": ("cough", "kaszel"),
    "Zyanose": ("cyanosis", "sinica"),
    "Schwindel": ("dizziness", "zawroty głowy"),
    "Sturz": ("fall", "upadek"),
    "Kopfverletzung": ("head injury", "uraz głowy"),
    "Hämatom": ("bruise", "siniak"),
    "Rötung": ("redness", "zaczerwienienie"),
    "Ödem": ("swelling", "obrzęk"),
    "Dekubitus": ("pressure ulcer", "odleżyna"),
    "Hautriss": ("skin tear", "rozdarcie skóry"),
    "Hautdefekt": ("open skin", "otwarta rana"),
    "Blutung": ("bleeding", "krwawienie"),
    "Nasenbluten": ("nosebleed", "krwawienie z nosa"),
    "Infektionszeichen": ("signs of infection", "oznaki infekcji"),
    "Augensekret": ("eye discharge", "wydzielina z oka"),
    "Verwirrtheit": ("confusion", "dezorientacja"),
    "Unruhe": ("restlessness", "niepokój"),
    "Angst": ("anxiety", "lęk"),
    "Beruhigung": ("calmer", "uspokojenie"),
    "Schlaf": ("sleep", "sen"),
    "Schwitzen": ("sweating", "pocenie się"),
    "Schüttelfrost": ("chills", "dreszcze"),
    "Bewusstseinsstörung": ("impaired consciousness", "zaburzenia świadomości"),
    "Schwäche": ("weakness", "osłabienie"),
    "verwaschene Sprache": ("slurred speech", "niewyraźna mowa"),
    "Schluckbeschwerden": ("difficulty swallowing", "problemy z przełykaniem"),
    "geringe Nahrungsaufnahme": ("eating little", "mało je"),
    "trüber Urin": ("cloudy urine", "mętny mocz"),
    "übelriechender Urin": ("foul-smelling urine", "nieprzyjemny zapach moczu"),
    "weicher Bauch": ("soft abdomen", "miękki brzuch"),
    "Bewegungseinschränkung": ("restricted movement", "ograniczenie ruchomości"),
    "nächtliches Aufstehen": ("getting up at night", "wstawanie w nocy"),
    "Beschwerden": ("complaints", "dolegliwości"),
    "Müdigkeit": ("tiredness", "zmęczenie"),
    "Juckreiz": ("itching", "świąd"),
    "Brustschmerzen": ("chest pain", "ból w klatce piersiowej"),
    "Krampfanfall": ("seizure", "napad drgawek"),
    # care actions
    "Duschen": ("shower", "prysznic"),
    "Körperpflege": ("body care", "toaleta ciała"),
    "Gesichtswäsche": ("face wash", "mycie twarzy"),
    "Mundpflege": ("mouth care", "higiena jamy ustnej"),
    "Verbandswechsel": ("dressing change", "zmiana opatrunku"),
    "Wundversorgung": ("wound care", "opatrzenie rany"),
    "Lagerung": ("repositioning", "zmiana pozycji"),
    "Mobilisation": ("mobilisation", "mobilizacja"),
    "Fersenschutz": ("heel protector", "ochraniacz pięt"),
    "Hüftprotektor": ("hip protector", "ochraniacz bioder"),
    "Inkontinenzversorgung": ("incontinence care", "zmiana pieluchomajtek"),
    "Getränke andicken": ("thickened drinks", "zagęszczanie napojów"),
    "Kühlung": ("cooling", "chłodzenie"),
    "Hausarzt informieren": ("inform the GP", "poinformowanie lekarza rodzinnego"),
    "Hausarztvisite": ("GP visit", "wizyta lekarza rodzinnego"),
    "Notruf": ("emergency call", "wezwanie pogotowia"),
    "Urinprobe": ("urine sample", "próbka moczu"),
    "Frühstück": ("breakfast", "śniadanie"),
    "Mittagessen": ("lunch", "obiad"),
    "Abendessen": ("dinner", "kolacja"),
    "Angehörige informieren": ("inform relatives", "poinformowanie rodziny"),
    "Toilettengang": ("toilet visit", "wyjście do toalety"),
    "Kompressionsstrümpfe": ("compression stockings", "pończochy uciskowe"),
    # sites
    "Knie": ("knee", "kolano"),
    "Hüfte": ("hip", "biodro"),
    "Ferse": ("heel", "pięta"),
    "Sakralbereich": ("sacral area", "okolica krzyżowa"),
    "Sitzbein": ("ischium", "kość kulszowa"),
    "Lendenwirbelsäule": ("lower back", "odcinek lędźwiowy"),
    "Unterschenkel": ("lower leg", "podudzie"),
    "Oberarm": ("upper arm", "ramię"),
    "Unterarm": ("forearm", "przedramię"),
    "Arm": ("arm", "ręka"),
    "Auge": ("eye", "oko"),
    "Rücken": ("back", "plecy"),
    "Kopf": ("head", "głowa"),
    "Bauch": ("abdomen", "brzuch"),
    "Haut": ("skin", "skóra"),
    "Fuß": ("foot", "stopa"),
    "Hand": ("hand", "dłoń"),
    # intake and scales
    "Trinkmenge": ("fluids", "płyny"),
    "Orangensaft": ("orange juice", "sok pomarańczowy"),
    "Tee": ("tea", "herbata"),
    "Trinknahrung": ("nutritional drink", "napój odżywczy"),
    "NRS": ("pain score", "skala bólu"),
    "Braden": ("Braden score", "skala Braden"),
    "Dekubitus-Kategorie": ("pressure ulcer category", "kategoria odleżyny"),
}

QUANTITIES = {
    "blood_pressure": ("Blood pressure", "Ciśnienie"),
    "pulse": ("Pulse", "Tętno"),
    "temperature": ("Temperature", "Temperatura"),
    "blood_glucose": ("Blood sugar", "Cukier"),
    "oxygen_saturation": ("Oxygen saturation", "Saturacja"),
    "respiratory_rate": ("Breathing rate", "Oddechy"),
    "weight": ("Weight", "Waga"),
    "weight_gain": ("Weight gain", "Przyrost wagi"),
    "weight_loss": ("Weight loss", "Utrata wagi"),
    "wound_size": ("Wound size", "Wielkość rany"),
    "urine_output": ("Urine output", "Ilość moczu"),
}

WORDS = {
    "present": ("yes", "tak"),
    "absent": ("no", "nie"),
    "possible": ("possibly", "możliwe"),
    "left": ("left side", "strona lewa"),
    "right": ("right side", "strona prawa"),
    "both": ("both sides", "obustronnie"),
    "given": ("given", "podano"),
    "refused": ("refused", "odmowa"),
    "withheld": ("not given", "nie podano"),
    "planned": ("planned", "zaplanowano"),
    "done": ("done", "wykonano"),
    "not_done": ("not done", "nie wykonano"),
    "as_needed": ("as needed", "doraźnie"),
    "resident": ("Resident", "Mieszkaniec"),
    "room": ("room", "pokój"),
    "times": ("times", "razy"),
    "german_term": ("German term", "termin niemiecki"),
    "confirm": (
        "Please confirm what you dictated:",
        "Proszę potwierdzić, co Pan/Pani podyktował(a):",
    ),
    "check": ("Please check the German note:", "Proszę sprawdzić niemiecką notatkę:"),
    "omitted": ("Missing from the German note", "Brakuje w niemieckiej notatce"),
    "added": ("In the German note but not dictated", "W niemieckiej notatce, ale nie podyktowano"),
    "changed": (
        "Recorded differently in the German note",
        "Zapisano inaczej w niemieckiej notatce",
    ),
}

# Unit words that are German; symbols (mg, ml, °C, mmHg) need no translation.
UNIT_WORDS = {
    "Hub": ("puffs", "wdechy"),
    "Tropfen": ("drops", "krople"),
    "Beutel": ("sachet", "saszetka"),
    "IE": ("units", "jednostki"),
}

_LANGUAGES = {"en": 0, "pl": 1}
_CLOCK_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")


def _word(key: str, language: str) -> str:
    return WORDS[key][_LANGUAGES[language]]


def _term(german: str | None, category: str, language: str) -> str:
    if german is None:
        return ""
    term = canonical(category, german)
    if term in TERMS:
        return TERMS[term][_LANGUAGES[language]]
    return f"„{german}“ ({_word('german_term', language)})"


def _capitalized(text: str) -> str:
    # str.capitalize() would lowercase the rest, including German nouns and "GP".
    return text[:1].upper() + text[1:]


def _unit(unit: str | None, language: str) -> str:
    if unit is None:
        return ""
    term = canonical("unit", unit)
    return UNIT_WORDS[term][_LANGUAGES[language]] if term in UNIT_WORDS else unit


def _number(value: float, language: str) -> str:
    text = f"{value:g}"
    return text.replace(".", ",") if language == "pl" else text


def _side_and_site(site: str | None, laterality, language: str) -> str:
    parts = []
    if site and canonical_site(site):
        parts.append(_term(canonical_site(site), "site", language))
    if laterality is not None:
        parts.append(_word(str(laterality), language))
    return ", ".join(parts)


def _time(context: str | None) -> str:
    # Free-text context is German; only clock times are safe to show untranslated.
    match = _CLOCK_TIME.search(context) if context else None
    return f" ({match.group()})" if match else ""


def render_fact(fact: Fact, language: str) -> str:
    match fact:
        case ResidentRef():
            room = f", {_word('room', language)} {fact.room}" if fact.room else ""
            return f"{_word('resident', language)}: {fact.name}{room}"
        case Measurement():
            label = QUANTITIES[fact.quantity][_LANGUAGES[language]]
            values = "/".join(_number(v, language) for v in fact.values)
            return f"{label}: {values} {_unit(fact.unit, language)}{_time(fact.context)}"
        case Medication():
            dose = ""
            if fact.dose is not None:
                dose = f" {_number(fact.dose, language)} {_unit(fact.unit, language)}".rstrip()
            details = [_word(fact.status, language)]
            if fact.as_needed:
                details.append(_word("as_needed", language))
            if fact.route:
                details.append(fact.route)
            side = _side_and_site(fact.site, fact.laterality, language)
            if side:
                details.append(side)
            return f"{fact.drug}{dose}: {', '.join(details)}{_time(fact.context)}"
        case Finding():
            concept, polarity = canonical_finding(fact.concept, fact.polarity)
            # Unknown concepts come back folded; show those as they were written.
            shown = concept if concept in TERMS else fact.concept
            name = _capitalized(_term(shown, "finding", language))
            line = f"{name}: {_word(polarity, language)}"
            side = _side_and_site(fact.site, fact.laterality, language)
            if side:
                line += f" — {side}"
            if fact.count and fact.count > 1:
                line += f" ({fact.count} {_word('times', language)})"
            return line + _time(fact.context)
        case CareMeasure():
            action = _capitalized(_term(fact.action, "action", language))
            return f"{action}: {_word(fact.status, language)}{_time(fact.context)}"
        case Intake():
            substance = _capitalized(_term(fact.substance, "substance", language))
            return f"{substance}: {_number(fact.amount, language)} {_unit(fact.unit, language)}"
        case Score():
            maximum = f"/{_number(fact.maximum, language)}" if fact.maximum else ""
            label = _capitalized(_term(fact.scale, "scale", language))
            return f"{label}: {_number(fact.value, language)}{maximum}"


def render_problem(discrepancy: Discrepancy, language: str) -> str:
    """One line per problem, describing the dictated fact (or the added one)."""
    fact = discrepancy.source if discrepancy.type != "added" else discrepancy.output
    heading = _word(discrepancy.type, language)
    if fact is None:
        return f"{heading}: {discrepancy.detail}"
    if discrepancy.type == "changed" and discrepancy.output is not None:
        after = render_fact(discrepancy.output, language)
        return f"{heading}: {render_fact(fact, language)} → {after}"
    return f"{heading}: {render_fact(fact, language)}"


def render_checklist(
    source_facts: list[Fact],
    problems: list[Discrepancy],
    language: str,
    narrative_findings: tuple[dict, ...] = (),
) -> str:
    lines = [_word("confirm", language)]
    lines += [f"  - {render_fact(fact, language)}" for fact in source_facts]
    serious = [d for d in problems if d.severity != "minor"]
    if serious or narrative_findings:
        lines.append(_word("check", language))
        lines += [f"  ! {render_problem(d, language)}" for d in serious]
        # Narrative findings are quotes from the German note; they stay in German.
        lines += [
            f"  ! {_word('added', language)}: „{f['statement']}“"
            for f in narrative_findings
            if f.get("statement")
        ]
    return "\n".join(lines)
