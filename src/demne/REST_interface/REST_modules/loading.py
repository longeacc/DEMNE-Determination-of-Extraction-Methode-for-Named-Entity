# pylint: disable=wildcard-import,unused-wildcard-import
from .calculs import *
from .categorization import create_ban_words_tfidf
from .extraction import *


def load_data_annotations(file_path):

    # 1 - Extraction + Stemming of the data
    docs = load_from_brat(file_path, merge_all_fragments=True)
    annotations = extract_annotations(docs, need_translation=False)
    annotations1 = stemming(annotations)
    annotations2 = annotations1

    # 2 - Initialization of major variables
    path = file_path
    data, ent_cat, ban_words_entities = create_data(annotations2)
    df = pd.DataFrame(
        data, columns=["entity", "category", "text", "occurrences", "stems", "places"]
    )
    df_tf_results = calculate_tfidf(ent_cat, df)
    homogeneity_score = calculate_homogeneity_score(df, ent_cat, 10)
    ban_words_tfidf = create_ban_words_tfidf(ent_cat)
    if get_ent(ent_cat):
        current_entity = get_ent(ent_cat)[0]
    else:
        current_entity = None

    return (
        path,
        ent_cat,
        ban_words_entities,
        df,
        df_tf_results,
        homogeneity_score,
        ban_words_tfidf,
        current_entity,
    )


def load_json(path, df, homogeneity_score, ent_cat):

    (
        progress_ent_cat,
        progress_isnotfp,
        progress_isnotfn,
        progress_ban_words_entities,
        progress_df_results,
    ) = load_progress(path)
    result = {}

    if progress_ent_cat:
        result["ent_cat"] = progress_ent_cat
    if progress_isnotfp:
        result["list_isNotFP"] = progress_isnotfp
    if progress_isnotfn:
        result["list_isNotFN"] = progress_isnotfn
    if progress_ban_words_entities:
        result["ban_words_entities"] = progress_ban_words_entities
    if progress_df_results is not None and not progress_df_results.empty:
        result["df_results"] = progress_df_results
    else:
        result["df_results"] = initiate_df_results(
            df, homogeneity_score, result.get("ent_cat", ent_cat)
        )

    return result
