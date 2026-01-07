from cv_pipeline import cv_processing_pipeline
from offer_pipeline import offer_processing_pipeline
from comparision import score_calculation


def pipeline_test(cv_path_list, offer):
    cv_profiles = []
    # Process the job offer
    offer_profile, offer_tokens = offer_processing_pipeline(offer)
    print(f"\nOffer processed using {offer_tokens} tokens.")

    for cv_path in cv_path_list:
        cv_profile, cv_tokens = cv_processing_pipeline(cv_path)
        cv_profiles.append((cv_path, cv_profile))


    # Compare each CV against the job offer
    for cv_path, cv_profile in cv_profiles:
        score = score_calculation(cv_profile, offer_profile)
        print(f"\nCV: {cv_path} \nScore: {score:.4f}")


if __name__ == "__main__":
    job_offer_text = """
        About the job
    Cognite operates at the forefront of industrial digitalization, building AI and data solutions that solve some of the world’s hardest, highest-impact problems. With unmatched industrial heritage and a comprehensive suite of AI capabilities, including low-code AI agents, Cognite accelerates the digital transformation to drive operational improvements.

    Our moonshot is bold: unlock $100B in customer value by 2035 and redefine how global industry works.

    What Cognite is Relentless to achieve

    We thrive in challenges. We challenge assumptions. We execute with speed and ownership. If you view obstacles as signals to step forward - not step back - you’ll feel at home here. Join us in this venture where AI and data meet ingenuity, and together, we forge the path to a smarter, more connected industrial future.

    How you’ll demonstrate Ownership

    We're looking for a Data Engineer who is ready to tackle big challenges and advance their career. In this role, you'll join a dynamic team committed to creating cutting-edge solutions that significantly impact critical industries such as Power & Utilities, Energy, and Manufacturing. You'll collaborate with industry leaders, solution architects, data scientists and project managers, all dedicated to deploying and optimizing digital solutions that empower our clients to make informed business decisions.

    Lead the design and implementation of scalable and efficient data engineering solutions using our platform Cognite Data Fusion®.
    Drive and manage integrations, extractions, data modeling, and analysis using Cognite data connectors and SQL, Python/Java and Rest APIs.
    Create custom data models for data discovery, mapping, and cleansing.
    Collaborate with data scientists, project managers and solution architects engineers on project deliveries to enable our customers to achieve the full potential of our industrial dataops platform.
    Conduct code reviews and implement best practices to ensure high-quality and maintainable code and deliveries.
    Support customers and partners in conducting data engineering tasks with Cognite products.
    Contribute to the development of Cognite’s official tools and SDKs.
    Collaborate with our Engineering and Product Management teams to turn customer needs into a prioritized pipeline of product offerings.

    The Impact you bring to Cognite

    Have a DevOps mindset, and experience with Git, CI/CD, deployment environments
    Enjoys working in cross-functional teams
    Able to independently investigate and solve problems
    Humility to ask for help and enjoy sharing knowledge with others

    Required Qualifications

    Minimum 3-5 years of relevant experience in a customer-facing Data intense role
    Experience delivering production-grade data pipelines using e.g. Python, SQL and Rest APIs
    Experience with distributed computing such as Kubernetes and managed cloud services such as GCP and/or Azure

    Preferred Experience

    Bachelor or Master degree in computer science or similar. Relevant experience can compensate for formal education

    Equal Opportunity

    Cognite is committed to creating a diverse and inclusive environment at work and is proud to be an equal opportunity employer. All qualified applicants will receive the same level of consideration for employment.


        """

    cv_folder_path = "data/training"

    # extracting cv paths from the folder
    import os
    cv_paths = [os.path.join(cv_folder_path, f) for f in os.listdir(cv_folder_path)]
    pipeline_test(cv_paths, job_offer_text)

