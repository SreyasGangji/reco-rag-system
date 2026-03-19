class ExplanationService:
    def build_explanation(
        self,
        title: str,
        predicted_rating: float,
        similar_context: list[str]
    ) -> str:
        if similar_context:
            if len(similar_context) == 1:
                context_text = similar_context[0]
            else:
                context_text = " and ".join(similar_context[:2])

            return (
                f"Recommended because {title} aligns with your predicted preferences "
                f"(predicted rating: {predicted_rating:.2f}) and is semantically related "
                f"to titles such as {context_text}."
            )

        return (
            f"Recommended because {title} aligns with your predicted preferences "
            f"with a predicted rating of {predicted_rating:.2f}."
        )


explanation_service = ExplanationService()