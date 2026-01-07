import os
import json
from dotenv import load_dotenv
from mistralai import Mistral


load_dotenv()


API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY)


PROMPT = """
Convert all text to english from any language.
Extract information from this job description and return ONLY a JSON object.

Return these 5 sections:
1. Education - list of required degrees/qualifications
2. Experience - list of required work experience 
3. Location - a dictionary with "city" and "country" as separate fields
4. Remote - "yes" if job is remote/work from home, "no" if not remote
5. Skills - list of all required skills

Format your response as pure JSON, no explanation, no markdown.

Example for Location:
{"city": "Paris", "country": "France"}
or
{"city": "", "country": ""} if not found
"""


def extract_job_info(job_description_text):

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": job_description_text}
        ],
        temperature=0
    )

    ai_response = response.choices[0].message.content


    start = ai_response.find("{")
    end = ai_response.rfind("}")
    json_text = ai_response[start:end + 1]


    job_info = json.loads(json_text)


    tokens_used = response.usage.total_tokens

    return job_info, tokens_used



if __name__ == "__main__":

    example_job = """About the job
Assistant Pricing : Etude des données marché (parts de marchés, structure des segments, études ad-oc), Initialisation des études de positionnements pricing (analyse contenu produit, suivi promo), benchmark concurrence (stratégie, go to market etc…)

Data Management : Collecte et organisation des données (Rassembler des données issues de différentes sources (ventes, campagnes marketing, réseaux sociaux, etc.), vérification (s'assurer de la qualité, de la cohérence et de la fiabilité des données), analyse (statistiques), reporting (créer des tableaux de bord et des rapports pour aider à la prise de décision) et support aux équipes marketing (optimiser la performance et les stratégies marketing)
its a remote job from villjueif.
 Profil 

Formation niveau Bac +5, Étudiant(e) en école de commerce, d'ingénieur ou université avec une spécialisation en marketing, data, ou business intelligence, vous êtes à l'aise avec les chiffres et les outils d'analyse. Vous êtes rigoureux(se), curieux(se) et avez un bon esprit d'analyse.

Compétences Clés

Bonne maîtrise d'Excel (TCD etc…), Power BI ou d'outils de data visualisation

Connaissances en bases de données (SQL est un plus)

Sensibilité marketing et compréhension des enjeux business

Capacité à structurer, nettoyer et analyser des données

Esprit d'équipe et autonomie

Intérêt pour le digital et la data marketing

Organisation et gestion des priorités

Anglais courant impératif, la maitrise d'une langue parlée dans la région (arabe, turc) serait un fort atout

Chez Stellantis, nous évaluons les candidats selon leurs

qualifications, leurs mérites et les besoins du métier.

Nous accueillons les candidatures des personnes de tout

genre, âge, ethnie, nationalité, religion, orientation sexuelle,

et handicap. La diversité de nos équipes nous permettra de

mieux appréhender l'évolution des besoins de nos clients et

de notre environnement futur.

 Durée du contrat 

6 mois

At Stellantis, we assess candidates based on qualifications, merit and business needs. We welcome applications from people of all gender identities, age, ethnicity, nationality, religion, sexual orientation and disability. Diverse teams will allow us to better meet the evolving needs of our customers and care for our future.
    
    """


    result, tokens = extract_job_info(example_job)


    print("\n=== JOB INFORMATION ===")
    print(json.dumps(result, indent=2))
    print(f"\nTokens used: {tokens}")