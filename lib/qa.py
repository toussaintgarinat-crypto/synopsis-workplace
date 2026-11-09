def build_qa_prompt(titre, resume, question):
    return (
        "Réponds uniquement à partir du résumé ci-dessous. Si l'information n'y est pas, dis-le.\n\n"
        f"Vidéo : {titre}\n\nRésumé :\n{resume}\n\nQuestion : {question}\n"
    )


def answer_question(titre, resume, question, llm):
    return llm(build_qa_prompt(titre, resume, question)).strip()
