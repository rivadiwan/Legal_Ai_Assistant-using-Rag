def get_case_chunks(
    case_name,
    chunks,
):

    case_chunks = []

    for chunk in chunks:

        if (
            chunk["case_name"].lower()
            ==
            case_name.lower()
        ):

            case_chunks.append(chunk)

    return case_chunks
