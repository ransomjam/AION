"""cyber_reasoning.py — structured threat-reasoning examples for AION-Cyber.

Reference material teaches a model *what* threats are.  This module teaches it
*how to work one out*: read a message, name the observable indicators, reach a
category, assign a risk, and recommend an action that a real person can take.

Every example has the same skeleton::

    Channel / Sender / Message
    Indicators observed:   the evidence, one per line
    Threat category:       what kind of attack this is
    Risk:                  High / Medium / Low / None
    Assessment:            how the indicators compose into the verdict
    Recommendation:        what the recipient should do

The skeleton never varies, because the skeleton is the lesson.  A model that
has seen thousands of these learns the *shape* of an analysis and can apply it
to a message no one wrote down in advance.

Safe examples are not filler
----------------------------
Roughly a third of what this module generates is legitimate traffic, reasoned
through in the identical format and ending in "Risk: None".  A corpus of
nothing but scams teaches a model that every message is a scam, which is worse
than useless in production: false positives are how a security tool loses the
user's trust and gets ignored precisely when it is right.  The safe examples
carry the harder lesson, that a bank message mentioning a code, an amount, and
an app is *normal*, and that what makes the scam a scam is the request to act
on a credential through an unofficial channel.

Provenance
----------
Everything here is ``synthetic`` and is labelled so in the manifest.  These are
constructed examples, not intercepted messages.  Real messages contributed by
real users are a separate pipeline with a consent and anonymisation path, and
they must never be silently mixed with generated text — the second operating
invariant exists precisely so a later reader can tell which is which.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

# ── the local threat landscape ────────────────────────────────────────────────
# Cameroonian digital life, because that is who the first users are.  A model
# trained only on US-centric phishing has never seen "MoMo", "FCFA", or a
# message that switches between French and English mid-sentence.


@dataclass(frozen=True)
class Brand:
    name: str
    kind: str            # money | bank | telecom | platform | logistics
    lookalike: str       # the domain an attacker would register
    official: str        # what the real channel actually is


BRANDS: tuple[Brand, ...] = (
    Brand("MTN MoMo", "money", "momo-secure.net", "the MTN MoMo app or 126"),
    Brand("Orange Money", "money", "orange-verify.net", "the Orange Money app or #150#"),
    Brand("Express Union", "money", "expressunion-online.net", "an Express Union branch"),
    Brand("Ecobank", "bank", "ecobank-alert.net", "the Ecobank app or your branch"),
    Brand("Afriland First Bank", "bank", "afriland-secure.co", "the Afriland app or your branch"),
    Brand("UBA", "bank", "uba-verify.net", "the UBA app or your branch"),
    Brand("Société Générale", "bank", "socgen-cm-verify.net", "the bank's official app"),
    Brand("Camtel", "telecom", "camtel-billing.net", "the Camtel customer service line"),
    Brand("Nexttel", "telecom", "nexttel-update.net", "the Nexttel customer service line"),
    Brand("WhatsApp", "platform", "whatsapp-verify.co", "the app's own settings screen"),
    Brand("Facebook", "platform", "facebook-security.co", "the app's own settings screen"),
    Brand("Gmail", "platform", "google-account-verify.net", "the Google account settings page"),
    Brand("DHL", "logistics", "dhl-cm-delivery.net", "the DHL tracking page"),
    Brand("Jumia", "platform", "jumia-refund.net", "the Jumia app"),
)

CHANNELS: tuple[str, ...] = ("SMS", "WhatsApp", "Email", "Messenger", "Marketplace")

CITIES: tuple[str, ...] = ("Douala", "Yaoundé", "Bamenda", "Buea", "Bafoussam",
                           "Garoua", "Limbe", "Kribi", "Ngaoundéré", "Maroua")

FIRST_NAMES: tuple[str, ...] = ("Jean", "Marie", "Paul", "Grace", "Emmanuel", "Sandrine",
                                "Eric", "Bernadette", "Serge", "Aïcha", "Brice", "Nadège")

AMOUNTS: tuple[str, ...] = ("5,000 FCFA", "15,000 FCFA", "25,000 FCFA", "50,000 FCFA",
                            "150,000 FCFA", "500,000 FCFA", "1,000,000 FCFA", "2,500,000 FCFA")

SHORTENERS: tuple[str, ...] = ("bit.ly/{slug}", "tinyurl.com/{slug}", "cutt.ly/{slug}")

SLUGS: tuple[str, ...] = ("verify-now", "claim237", "acct-check", "momo-fix",
                          "win-cash", "secure-login", "update-app")


def _phone(rng: random.Random) -> str:
    prefix = rng.choice(("650", "651", "654", "655", "670", "677", "690", "698"))
    return f"+237 {prefix} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}"


def _short_link(rng: random.Random) -> str:
    return rng.choice(SHORTENERS).format(slug=rng.choice(SLUGS))


# ── the example ───────────────────────────────────────────────────────────────

RISK_NONE, RISK_LOW, RISK_MEDIUM, RISK_HIGH = "None", "Low", "Medium", "High"


@dataclass
class ReasoningExample:
    """One worked analysis, rendered into the corpus's teaching format."""
    channel: str
    sender: str
    message: str
    indicators: list[str]
    category: str
    risk: str
    assessment: str
    recommendation: str
    language: str = "en"
    family: str = ""
    seed_of: str = ""          # non-empty when derived from a curated seed

    def render(self) -> str:
        lines = [
            "Message analysis",
            f"Channel: {self.channel}",
            f"Sender: {self.sender}",
            "Message:",
            self.message,
            "",
            "Indicators observed:",
        ]
        if self.indicators:
            lines += [f"- {indicator}" for indicator in self.indicators]
        else:
            lines.append("- No indicators of deception were found.")
        lines += [
            "",
            f"Threat category: {self.category}",
            f"Risk: {self.risk}",
            f"Assessment: {self.assessment}",
            f"Recommendation: {self.recommendation}",
        ]
        return "\n".join(lines)


# ── attack families ───────────────────────────────────────────────────────────
# Each family is a *way of reasoning*, not a string.  Surface text varies with
# the brand, channel, amount, and language; the indicator set and the logic
# tying indicators to a verdict stay fixed, which is precisely the invariance
# we want the model to extract.


@dataclass(frozen=True)
class AttackFamily:
    key: str
    category: str
    risk: str
    #: builds (message, indicators, assessment, recommendation) from a context
    build: object
    languages: tuple[str, ...] = ("en", "fr")
    channels: tuple[str, ...] = CHANNELS
    kinds: tuple[str, ...] = ()          # restrict to these brand kinds
    field_names: tuple[str, ...] = field(default_factory=tuple)


# A mobile-money account has a PIN; a mailbox has a password.  Getting the
# secret's name right per brand matters more than it looks: the model should
# learn that the tell is "the message asks for the secret", not the word "PIN".
_SECRET = {"money": ("PIN", "code PIN"), "bank": ("PIN", "code PIN"),
           "telecom": ("PIN", "code PIN"), "platform": ("password", "mot de passe"),
           "logistics": ("password", "mot de passe")}


def _credential_harvest(ctx: dict) -> tuple[str, list[str], str, str]:
    brand, link, lang = ctx["brand"], ctx["link"], ctx["lang"]
    secret_en, secret_fr = _SECRET.get(brand.kind, ("password", "mot de passe"))
    secret = secret_fr if lang == "fr" else secret_en
    if lang == "fr":
        message = (f"{brand.name}: votre compte a été suspendu. "
                   f"Confirmez votre {secret} immédiatement pour éviter le blocage: {link}")
    else:
        message = (f"{brand.name}: your account has been suspended. "
                   f"Confirm your {secret} immediately to avoid being blocked: {link}")
    indicators = [
        f"Brand impersonation: the message claims to be {brand.name}.",
        f"Credential request: it asks for the account {secret_en}, which no provider ever requests.",
        "Urgency and threat: suspension is used to shorten the recipient's thinking time.",
        f"Unofficial destination: the link points to {link}, not to a domain the brand controls.",
        "Channel mismatch: an account action arrives by message rather than inside the app.",
    ]
    assessment = (
        f"The single decisive indicator is the credential request. A provider can "
        f"already act on the account and has no reason to ask for the {secret_en}, so "
        f"any message that does is impersonating one. The urgency and the unofficial "
        f"link both support that reading rather than establishing it independently."
    )
    recommendation = (
        f"Do not open the link and do not enter the {secret_en}. If there is any doubt "
        f"about the account, check it through {brand.official}."
    )
    return message, indicators, assessment, recommendation


def _advance_fee(ctx: dict) -> tuple[str, list[str], str, str]:
    prize, fee, lang = ctx["amount"], ctx["small_amount"], ctx["lang"]
    brand = ctx["brand"]
    if lang == "fr":
        message = (f"Félicitations! Vous avez gagné {prize} à la tombola {brand.name}. "
                   f"Payez des frais d'activation de {fee} pour recevoir votre prix.")
    else:
        message = (f"Congratulations! You have won {prize} in the {brand.name} draw. "
                   f"Pay an activation fee of {fee} to receive your prize.")
    indicators = [
        f"Unsolicited winnings: a prize of {prize} from a draw the recipient never entered.",
        f"Advance fee: payment of {fee} is demanded before anything is received.",
        "Payment direction is reversed: a genuine payout never requires the recipient to send money first.",
        f"Brand borrowing: {brand.name} is invoked to supply credibility the sender does not have.",
    ]
    assessment = (
        "The structure identifies this regardless of the amounts involved: value is "
        "promised, a smaller payment is demanded first, and the smaller payment is "
        "the only real transaction. The prize does not exist, so the fee is the "
        "entire scheme."
    )
    recommendation = (
        "Send nothing. A real prize is never unlocked by a payment from the winner. "
        "Delete the message and block the sender."
    )
    return message, indicators, assessment, recommendation


def _fake_support(ctx: dict) -> tuple[str, list[str], str, str]:
    brand, link, lang = ctx["brand"], ctx["link"], ctx["lang"]
    if lang == "fr":
        message = (f"Bonjour, je suis du service client {brand.name}. Nous avons détecté "
                   f"un problème sur votre compte. Vérifiez vos identifiants ici: {link}")
    else:
        message = (f"Hello, I am from {brand.name} customer care. We detected a problem "
                   f"with your account. Please verify your credentials here: {link}")
    indicators = [
        f"Support impersonation: the sender claims to represent {brand.name}.",
        "Sender identity does not match the claim: it is an ordinary personal number, not a service line.",
        "Credential request routed outside the official application.",
        f"Unofficial destination: {link}.",
    ]
    assessment = (
        "Real support contacts a customer through the channel that already "
        "identifies them, and does not need credentials to see an account it "
        "administers. A support claim arriving from a personal number and asking "
        "for credentials inverts both facts."
    )
    recommendation = (
        f"Do not reply and do not follow the link. Contact {brand.official} directly "
        f"using a number you looked up yourself."
    )
    return message, indicators, assessment, recommendation


def _job_scam(ctx: dict) -> tuple[str, list[str], str, str]:
    pay, fee, lang = ctx["amount"], ctx["small_amount"], ctx["lang"]
    if lang == "fr":
        message = (f"Recrutement URGENT! Gagnez {pay} par semaine depuis votre téléphone. "
                   f"Frais d'inscription {fee} requis pour recevoir votre kit de démarrage.")
    else:
        message = (f"URGENT recruitment! Earn {pay} weekly working from your phone. "
                   f"Registration fee of {fee} required to receive your starter kit.")
    indicators = [
        f"Implausible pay: {pay} per week for unspecified phone work.",
        f"Up-front fee: {fee} is demanded from the applicant.",
        "No named employer, role, or place of work.",
        "Manufactured urgency in the opening word.",
    ]
    assessment = (
        "An employer pays the worker. A recruitment offer that collects money from "
        "the applicant has reversed the direction of employment, and the vagueness "
        "about the actual job is what makes the fee collectable from many people at "
        "once."
    )
    recommendation = (
        "Do not pay the registration fee. A legitimate employer never charges to "
        "hire. Verify any company through an address and a listed telephone number."
    )
    return message, indicators, assessment, recommendation


def _delivery_fee(ctx: dict) -> tuple[str, list[str], str, str]:
    brand, link, city, fee, lang = (ctx["brand"], ctx["link"], ctx["city"],
                                    ctx["small_amount"], ctx["lang"])
    if lang == "fr":
        message = (f"Votre colis {brand.name} est bloqué à la douane de {city}. "
                   f"Payez {fee} de frais de livraison pour le recevoir: {link}")
    else:
        message = (f"Your {brand.name} parcel is on hold at {city} customs. "
                   f"Pay a {fee} delivery fee to release it: {link}")
    indicators = [
        "Parcel pretext: it relies on the recipient expecting some delivery.",
        f"Payment demanded through a link rather than a carrier's own system: {link}.",
        "Link shortener or lookalike domain hides the true destination.",
        f"Small, plausible amount ({fee}) chosen to be paid without much thought.",
    ]
    assessment = (
        "The amount is deliberately small: the scheme profits from volume and from "
        "the payment card details entered on the page, not from the fee itself. "
        "Carriers bill through their own tracking systems, never through a "
        "shortened link in a message."
    )
    recommendation = (
        f"Do not pay through the link. Track the parcel through {brand.official} using "
        f"the tracking number from the original order."
    )
    return message, indicators, assessment, recommendation


def _malware_lure(ctx: dict) -> tuple[str, list[str], str, str]:
    brand, link, lang = ctx["brand"], ctx["link"], ctx["lang"]
    if lang == "fr":
        message = (f"Votre application {brand.name} est obsolète. Téléchargez la "
                   f"nouvelle version APK ici: {link}. Installez immédiatement.")
    else:
        message = (f"Your {brand.name} app is outdated. Download the new version APK "
                   f"here: {link}. Install immediately to avoid suspension.")
    indicators = [
        "Application package delivered outside the official store.",
        f"Download destination is attacker-controlled: {link}.",
        "Urgency paired with a consequence for not installing.",
        "An APK sideload asks the user to disable a protection, not just to click.",
    ]
    assessment = (
        "Legitimate applications update through the store that signed them. A "
        "message steering a user to install a package from elsewhere is asking for "
        "code execution on the device, which is a materially larger ask than a "
        "phishing page and is why this is treated as high risk even without a "
        "credential request."
    )
    recommendation = (
        "Do not download or install the file. Update applications only from the "
        "Play Store, and check the installed version there."
    )
    return message, indicators, assessment, recommendation


def _overpayment(ctx: dict) -> tuple[str, list[str], str, str]:
    extra, fee, lang = ctx["small_amount"], ctx["smaller_amount"], ctx["lang"]
    item = ctx["item"]
    if lang == "fr":
        message = (f"Je veux acheter votre {item} maintenant! Je paierai même {extra} de plus. "
                   f"Payez d'abord {fee} à mon agent de transport pour recevoir le paiement complet.")
    else:
        message = (f"I want to buy your {item} right now! I will even pay {extra} extra. "
                   f"Just pay {fee} to my transport agent first to receive the full payment.")
    indicators = [
        f"Unprompted overpayment of {extra} above the asking price.",
        f"Seller is asked to send {fee} before receiving anything.",
        "A third party ('transport agent') is introduced to justify the outgoing payment.",
        "Pressure to complete immediately, before the seller can check anything.",
    ]
    assessment = (
        "Overpayment is the bait and the agent fee is the theft. The offer of more "
        "than the asking price exists only to make the small outgoing payment feel "
        "safe; no buyer and no goods ever materialise."
    )
    recommendation = (
        "Never send money to receive a payment. Meet in a public place and confirm "
        "funds have actually arrived before handing over the item."
    )
    return message, indicators, assessment, recommendation


def _sim_swap(ctx: dict) -> tuple[str, list[str], str, str]:
    brand, lang = ctx["brand"], ctx["lang"]
    code = ctx["code"]
    if lang == "fr":
        message = (f"{brand.name}: code de transfert de ligne {code}. Communiquez ce code "
                   f"à notre agent pour finaliser la mise à jour de votre SIM.")
    else:
        message = (f"{brand.name}: line transfer code {code}. Share this code with our "
                   f"agent to complete your SIM update.")
    indicators = [
        "A one-time code is requested to be shared with a third party.",
        "The pretext is a SIM or line transfer, which would move the number to another device.",
        f"Brand impersonation: sent in the name of {brand.name}.",
        "A code the recipient did not request implies a transfer already in progress.",
    ]
    assessment = (
        "This is the setup for account takeover rather than an end in itself. "
        "Control of the number means control of every code sent to it, including "
        "mobile money and bank confirmations, so sharing the code hands over far "
        "more than the SIM."
    )
    recommendation = (
        f"Never share a one-time code with anyone, including someone claiming to be "
        f"staff. Report an unrequested transfer code to {brand.official} immediately."
    )
    return message, indicators, assessment, recommendation


def _relative_in_trouble(ctx: dict) -> tuple[str, list[str], str, str]:
    name, city, amount, lang = ctx["name"], ctx["city"], ctx["small_amount"], ctx["lang"]
    if lang == "fr":
        message = (f"C'est {name}. J'ai perdu mon téléphone, c'est mon nouveau numéro. "
                   f"Je suis bloqué à {city} et j'ai besoin de {amount} en urgence. Envoie par MoMo.")
    else:
        message = (f"It's {name}. I lost my phone, this is my new number. I am stuck in "
                   f"{city} and need {amount} urgently. Send it by MoMo.")
    indicators = [
        "Claimed identity arrives from an unknown number.",
        "The new-number explanation pre-empts the obvious check.",
        f"Immediate request for {amount} with a distressing reason.",
        "Emotional pressure discourages verification.",
    ]
    assessment = (
        "The new-number claim is load-bearing: it explains away the one thing that "
        "would otherwise expose the message. The distress is calibrated to make "
        "verifying feel like a delay the relative cannot afford."
    )
    recommendation = (
        "Call the person on their old, known number before sending anything, or ask "
        "something only they could answer. Do not send money to a number you cannot verify."
    )
    return message, indicators, assessment, recommendation


def _investment_scam(ctx: dict) -> tuple[str, list[str], str, str]:
    stake, lang = ctx["small_amount"], ctx["lang"]
    payout = ctx["amount"]
    if lang == "fr":
        message = (f"Investissement crypto garanti! Déposez {stake} et recevez {payout} "
                   f"en 48 heures. Places limitées, inscrivez-vous maintenant.")
    else:
        message = (f"Guaranteed crypto investment! Deposit {stake} and receive {payout} "
                   f"within 48 hours. Limited slots, register now.")
    indicators = [
        "Guaranteed return, which no genuine investment can offer.",
        f"Implausible multiple: {stake} becoming {payout} in 48 hours.",
        "Artificial scarcity to compress the decision.",
        "No named, regulated institution behind the offer.",
    ]
    assessment = (
        "The word 'guaranteed' paired with a short horizon is decisive on its own: "
        "return and risk cannot be separated, so an offer that removes the risk is "
        "misrepresenting the return. The deposit is the product."
    )
    recommendation = (
        "Do not deposit. Check any investment offer against a regulator's register "
        "before sending money, and treat guaranteed returns as proof of fraud."
    )
    return message, indicators, assessment, recommendation


# ── legitimate traffic, reasoned through identically ──────────────────────────

def _legit_receipt(ctx: dict) -> tuple[str, list[str], str, str]:
    brand, amount, name, lang = ctx["brand"], ctx["small_amount"], ctx["name"], ctx["lang"]
    ref = ctx["ref"]
    if lang == "fr":
        message = (f"Vous avez reçu {amount} de {name}. Nouveau solde: {ctx['amount']}. "
                   f"Réf: {ref}.")
    else:
        message = (f"You have received {amount} from {name}. New balance: {ctx['amount']}. "
                   f"Ref: {ref}.")
    indicators = [
        "Reports a completed transaction rather than requesting an action.",
        "No link, no credential request, no urgency.",
        "Includes a reference the recipient can check in the app.",
    ]
    assessment = (
        "A receipt is a statement of fact about something that already happened. "
        "There is nothing for the recipient to do and nothing for an attacker to "
        "gain, which is what distinguishes it from a message that merely mentions "
        "money."
    )
    recommendation = (
        f"No action needed. If the transaction looks unfamiliar, check the balance in "
        f"{brand.official} rather than replying to the message."
    )
    return message, indicators, assessment, recommendation


def _legit_otp(ctx: dict) -> tuple[str, list[str], str, str]:
    brand, code, lang = ctx["brand"], ctx["code"], ctx["lang"]
    if lang == "fr":
        message = (f"{brand.name}: votre code à usage unique est {code}. Ne le partagez "
                   f"avec personne. Il expire dans 5 minutes.")
    else:
        message = (f"{brand.name}: your one-time code is {code}. Do not share this code "
                   f"with anyone. It expires in 5 minutes.")
    indicators = [
        "Delivers a code rather than asking for one.",
        "Explicitly warns against sharing it.",
        "No link and no destination to visit.",
    ]
    assessment = (
        "The direction of the code is what matters. Delivering a code with a warning "
        "not to share it is the legitimate pattern; asking the recipient to read one "
        "back is the attack. Treating every message containing a code as suspicious "
        "would flag the security mechanism itself."
    )
    recommendation = (
        "Use the code only in the application that requested it. If no login was "
        "attempted, change the account password."
    )
    return message, indicators, assessment, recommendation


def _legit_notice(ctx: dict) -> tuple[str, list[str], str, str]:
    brand, lang = ctx["brand"], ctx["lang"]
    if lang == "fr":
        message = (f"Votre relevé mensuel {brand.name} est disponible dans l'application. "
                   f"Merci de votre confiance.")
    else:
        message = (f"Your monthly {brand.name} statement is ready in the app. "
                   f"Thank you for banking with us.")
    indicators = [
        "Points the recipient into the official application, not to an external link.",
        "Requests nothing and threatens nothing.",
        "Routine notice with no time pressure.",
    ]
    assessment = (
        "Directing a customer into the app they already have is the opposite of the "
        "phishing pattern, which must move them somewhere the attacker controls. "
        "The absence of a link is itself the reassuring signal."
    )
    recommendation = "No action needed. Open the app directly to read the statement."
    return message, indicators, assessment, recommendation


def _legit_personal(ctx: dict) -> tuple[str, list[str], str, str]:
    city, name, lang = ctx["city"], ctx["name"], ctx["lang"]
    item = ctx["item"]
    if lang == "fr":
        message = (f"Bonjour, est-ce que le {item} est toujours disponible? "
                   f"Je peux passer le voir demain matin à {city}.")
    else:
        message = (f"Hello, is the {item} still available? I can come and see it "
                   f"tomorrow morning in {city}.")
    indicators = [
        "Asks a question about the listing rather than proposing a payment.",
        "Proposes meeting in person in a named place.",
        "No link, no fee, no urgency.",
    ]
    assessment = (
        "A genuine buyer wants to see the item; a fraudulent one wants to move money "
        "before any meeting. Proposing an in-person viewing runs against every "
        "marketplace fraud pattern, which depends on distance."
    )
    recommendation = (
        f"Reply normally. Arrange to meet in a public place in {city} and accept "
        f"payment only once you are together."
    )
    return message, indicators, assessment, recommendation


FAMILIES: tuple[AttackFamily, ...] = (
    AttackFamily("credential_harvest", "Credential harvesting", RISK_HIGH, _credential_harvest,
                 kinds=("money", "bank", "platform", "telecom")),
    AttackFamily("advance_fee", "Advance-fee fraud", RISK_HIGH, _advance_fee),
    AttackFamily("fake_support", "Support impersonation", RISK_HIGH, _fake_support,
                 channels=("SMS", "WhatsApp", "Messenger"),
                 kinds=("money", "bank", "telecom", "platform")),
    AttackFamily("job_scam", "Employment fraud", RISK_HIGH, _job_scam,
                 channels=("WhatsApp", "Messenger", "SMS")),
    AttackFamily("delivery_fee", "Delivery-fee fraud", RISK_HIGH, _delivery_fee,
                 kinds=("logistics", "platform")),
    AttackFamily("malware_lure", "Malware delivery", RISK_HIGH, _malware_lure,
                 kinds=("money", "platform", "telecom")),
    AttackFamily("overpayment", "Marketplace overpayment fraud", RISK_HIGH, _overpayment,
                 channels=("Marketplace", "WhatsApp")),
    AttackFamily("sim_swap", "SIM-swap account takeover", RISK_HIGH, _sim_swap,
                 channels=("SMS",), kinds=("telecom", "money")),
    AttackFamily("relative_in_trouble", "Impersonation of a known contact", RISK_HIGH,
                 _relative_in_trouble, channels=("SMS", "WhatsApp")),
    AttackFamily("investment_scam", "Investment fraud", RISK_HIGH, _investment_scam,
                 channels=("WhatsApp", "Messenger", "SMS", "Email")),
    # Legitimate traffic — the false-positive lesson.
    AttackFamily("legit_receipt", "Legitimate transaction notice", RISK_NONE, _legit_receipt,
                 channels=("SMS",), kinds=("money", "bank")),
    AttackFamily("legit_otp", "Legitimate one-time code", RISK_NONE, _legit_otp,
                 channels=("SMS",), kinds=("bank", "money", "platform")),
    AttackFamily("legit_notice", "Legitimate service notice", RISK_NONE, _legit_notice,
                 channels=("SMS", "Email"), kinds=("bank", "money", "telecom")),
    AttackFamily("legit_personal", "Ordinary personal message", RISK_NONE, _legit_personal,
                 channels=("WhatsApp", "Marketplace", "Messenger")),
)

ITEMS: tuple[str, ...] = ("laptop", "phone", "fridge", "generator", "sofa", "bicycle")
ITEMS_FR: tuple[str, ...] = ("ordinateur", "téléphone", "réfrigérateur", "groupe électrogène",
                             "canapé", "vélo")


def _context(rng: random.Random, family: AttackFamily) -> dict:
    """Draw one concrete situation for a family to be expressed in."""
    pool = [b for b in BRANDS if not family.kinds or b.kind in family.kinds] or list(BRANDS)
    brand = rng.choice(pool)
    lang = rng.choice(family.languages)
    amounts = sorted(AMOUNTS, key=lambda a: int(a.split()[0].replace(",", "")))
    high = rng.choice(amounts[3:])
    low = rng.choice(amounts[:4])
    lower = rng.choice(amounts[:2])
    return {
        "brand": brand,
        "lang": lang,
        "link": rng.choice((brand.lookalike, _short_link(rng))),
        "amount": high,
        "small_amount": low,
        "smaller_amount": lower,
        "city": rng.choice(CITIES),
        "name": rng.choice(FIRST_NAMES),
        "code": f"{rng.randint(100000, 999999)}",
        "ref": f"TXN{rng.randint(100000, 999999)}",
        "item": rng.choice(ITEMS_FR if lang == "fr" else ITEMS),
    }


def _sender_for(rng: random.Random, family: AttackFamily, ctx: dict) -> str:
    """Who the message appears to come from.

    Scam families mostly use ordinary personal numbers while claiming to be an
    institution — the mismatch is itself an indicator, so the generated sender
    has to be consistent with the indicator list that names it.
    """
    brand = ctx["brand"]
    if family.risk == RISK_NONE:
        if family.key == "legit_personal":
            return f"{ctx['name']} — {ctx['city']}"
        return brand.name
    if family.key in ("fake_support", "relative_in_trouble", "job_scam", "investment_scam"):
        return _phone(rng)
    if family.key == "overpayment":
        return f"Buyer — Marketplace ({_phone(rng)})"
    return rng.choice((brand.name, f"{brand.name} Alert", _phone(rng)))


def generate(count: int, seed: int = 42) -> Iterator[ReasoningExample]:
    """Yield ``count`` examples, deterministically for a given ``seed``.

    Families are cycled rather than sampled so the mix is balanced by
    construction: an unlucky draw cannot leave a family with three examples and
    another with three hundred, and the safe-to-scam ratio holds exactly.
    """
    rng = random.Random(seed)
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        ctx = _context(rng, family)
        channel = rng.choice(family.channels)
        message, indicators, assessment, recommendation = family.build(ctx)
        yield ReasoningExample(
            channel=channel,
            sender=_sender_for(rng, family, ctx),
            message=message,
            indicators=indicators,
            category=family.category,
            risk=family.risk,
            assessment=assessment,
            recommendation=recommendation,
            language=ctx["lang"],
            family=family.key,
        )


# ── curated seeds ─────────────────────────────────────────────────────────────
# The Camtinel demo scenarios, which are hand-written and already validated
# against the shipped detection engine.  They anchor the generated set: if the
# templates ever drift into a style no real message has, these do not move.

CURATED: tuple[dict, ...] = (
    {
        "id": "sms_momo_pin", "channel": "SMS", "sender": "MTN MoMo Alert",
        "message": "Confirm your PIN to keep your account active or it will be blocked immediately.",
        "family": "credential_harvest", "kinds": ("money",),
    },
    {
        "id": "sms_lottery", "channel": "SMS", "sender": "Orange Money",
        "message": "Congratulations! You have won 500,000 FCFA. Claim your prize now: orange-lottery.cm",
        "family": "advance_fee", "kinds": ("money",),
    },
    {
        "id": "wa_support_impersonation", "channel": "WhatsApp", "sender": "+237 698 11 22 33",
        "message": ("Hello, I am from MTN customer care. We detected a problem with your MoMo "
                    "account. Please verify your PIN here: momo-secure.net"),
        "family": "fake_support", "kinds": ("money",),
    },
    {
        "id": "email_paypal_phish", "channel": "Email", "sender": "service@paypal-secure.net",
        "message": ("Your PayPal account has been suspended. Login to verify your password "
                    "within 24 hours: paypal-verify.net"),
        "family": "credential_harvest", "kinds": ("platform",),
    },
    {
        "id": "browser_apk_lure", "channel": "Other", "sender": "654 09 87 65",
        "message": ("Your Orange Money app is outdated. Download the new version APK here: "
                    "orange-update.net. Install immediately to avoid account suspension."),
        "family": "malware_lure", "kinds": ("money",),
    },
    {
        "id": "sms_receipt", "channel": "SMS", "sender": "MTN MoMo",
        "message": "You have received 15,000 FCFA from John D. New balance: 47,320 FCFA. Ref: TXN001.",
        "family": "legit_receipt", "kinds": ("money",),
    },
    {
        "id": "sms_otp_delivery", "channel": "SMS", "sender": "Ecobank",
        "message": "Your one time code is 483921. Do not share this code with anyone. It expires in 5 minutes.",
        "family": "legit_otp", "kinds": ("bank",),
    },
    {
        "id": "wa_family_message", "channel": "WhatsApp", "sender": "Maman",
        "message": "On se voit demain au marché vers 14h? N'oublie pas les tomates.",
        "family": "legit_personal", "kinds": (),
    },
)

_FAMILY_BY_KEY = {f.key: f for f in FAMILIES}


def curated_examples(seed: int = 42) -> Iterator[ReasoningExample]:
    """The hand-written scenarios, reasoned through with their family's logic."""
    rng = random.Random(seed)
    for entry in CURATED:
        family = _FAMILY_BY_KEY[entry["family"]]
        ctx = _context(rng, family)
        # A curated message keeps its own text; only the reasoning is generated,
        # so the analysis prose stays consistent with everything else.
        _, indicators, assessment, recommendation = family.build(ctx)
        yield ReasoningExample(
            channel=entry["channel"],
            sender=entry["sender"],
            message=entry["message"],
            indicators=indicators,
            category=family.category,
            risk=family.risk,
            assessment=assessment,
            recommendation=recommendation,
            language="fr" if entry["id"].endswith("_fr") else "en",
            family=family.key,
            seed_of=entry["id"],
        )


def family_summary() -> dict[str, int]:
    """How many families sit on each risk level — reported by the builder."""
    counts: dict[str, int] = {}
    for family in FAMILIES:
        counts[family.risk] = counts.get(family.risk, 0) + 1
    return counts
