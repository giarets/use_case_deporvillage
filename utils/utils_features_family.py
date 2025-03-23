import re
import pandas as pd


class ExtractFamilyFeatures:
    """
    A class to preprocess and extract features from item names strings.
    Using column 'family' it will generate features:
        - clothing_type: first word or phrase in the item name
        - sport_category: the rest of the item name
        - outlet_flag: 1 if the item name contains "Outlet", 0 otherwise
    """

    phrases_to_preserve = [
        "de montaña",
        "Trail Running",
        "Forros polares",
        "Ropa interior y térmica",
        "Tops y Sujetadores deportivos",
    ]

    def __init__(self):
        pass

    def __call__(self, df):
        return self._process_dataframe(df)

    def _process_dataframe(self, df):
        """
        Apply the string preprocessing pipeline to the 'family' column.
        """
        # Step 1: Find and preserve "y" phrases
        self._find_phrases_with_y(df["family"])

        # Step 2: Apply preprocessing to each row
        df[["clothing_type", "sport_category", "outlet_flag"]] = (
            df["family"].apply(self._preprocess_family_string).tolist()
        )
        return df

    @staticmethod
    def _find_phrases_with_y(strings):
        """
        Scans strings for words connected by "y" and adds them to the hardcoded
        phrases_to_preserve list.
        """
        new_phrases = set()
        for string in strings:
            matches = re.findall(r"\b\w+ y \w+\b", string)
            new_phrases.update(matches)

        # Extend the hardcoded phrases_to_preserve
        ExtractFamilyFeatures.phrases_to_preserve.extend(new_phrases)
        # Remove duplicates
        ExtractFamilyFeatures.phrases_to_preserve = list(
            set(ExtractFamilyFeatures.phrases_to_preserve)
        )

    @staticmethod
    def _preprocess_family_string(item_name):
        """
        String preprocessing pipeline for family names.
        """
        # Step 1: Remove "Outlet" and flag it
        outlet_flag = 1 if "Outlet" in item_name else 0
        item_name = item_name.replace("Outlet", "").strip()

        # Step 2: Remove "deportivos"
        item_name = re.sub(
            r"\bdeportivos\b", "", item_name, flags=re.IGNORECASE
        ).strip()

        # Step 3: Preserve predefined multi-word phrases
        for phrase in ExtractFamilyFeatures.phrases_to_preserve:
            item_name = re.sub(
                rf"\b{phrase}\b",
                phrase.replace(" ", "_"),
                item_name,
                flags=re.IGNORECASE,
            )

        # Step 4: Handle "y" connections (merge words connected by "y")
        item_name = re.sub(
            r"(\b\w+\b) y (\b\w+\b)", r"\1_y_\2", item_name, flags=re.IGNORECASE
        )

        # Step 5: Normalize text (convert to lowercase)
        item_name = item_name.lower()

        # Step 6: Split into words while treating preserved phrases as single units
        words = item_name.split()

        # Step 7: Extract clothing type and sport_category
        clothing_type = (
            words[0] if words else "unknown"
        )  # First word or preserved phrase is the clothing type
        sport_category = " ".join(words[1:]) if len(words) > 1 else "unknown"

        # Step 8: Restore preserved phrases to original form
        clothing_type = clothing_type.replace("_", " ")
        sport_category = sport_category.replace("_", " ")

        return clothing_type, sport_category, outlet_flag
