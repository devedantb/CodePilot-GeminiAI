import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import HtmlFormatter
import re

async def format_response(response):
    """
    Format the response from the LLM to include properly formatted code snippets
    and explanatory text.

    Args:
    - response (str): The raw response from the LLM.

    Returns:
    - str: The formatted response as HTML.
    """
    formatted_response = []
    code_block_pattern = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
    last_end = 0
    code_block_counter = 0

    for match in code_block_pattern.finditer(response):
        # Text before the code block
        text_before_code = response[last_end:match.start()]
        if text_before_code.strip():
            formatted_response.append(markdown.markdown(text_before_code.strip()))

        # The code block
        language = match.group(1) or 'text'
        code = match.group(2)
        try:
            lexer = get_lexer_by_name(language, stripall=True)
        except:
            lexer = TextLexer(stripall=True)
        formatter = HtmlFormatter()
        highlighted_code = highlight(code, lexer, formatter)
        code_block_counter += 1

        copy_button_html = f"""
            <button class="copy-button bg-grey-500 text-white py-1 px-3 rounded-md absolute top-2 right-2" data-code-block="code-block-{code_block_counter}">
                <span class="copied-message text-green-500 ml-2 hidden">Copied!</span>
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm">
                    <path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path>
                </svg>
            </button>
            
            <span style="color: #7dadff; font-weight: bold;">{language}</span>
            <div class="scrollable bg-gray-900 text-white p-4 rounded-md overflow-auto">
                {highlighted_code}
            </div>
            <textarea class="rounded-lg bg-gray-600" id="code-block-{code_block_counter}" style="position: absolute; left: -9999px;">{code}</textarea>
        """

        formatted_response.append(copy_button_html)

        last_end = match.end()

    # Any remaining text after the last code block
    if last_end < len(response):
        remaining_text = response[last_end:]
        if remaining_text.strip():
            formatted_response.append(markdown.markdown(remaining_text.strip()))

    return '\n'.join(formatted_response)