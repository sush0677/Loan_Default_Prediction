$(document).ready(function() {
    $('.custom-file-input').on('change', function() {
        // Extract and display the file name
        let fileName = $(this).val().split('\\').pop();
        $(this).siblings('.custom-file-label').addClass("selected").html(fileName);

        // Process file after a slight delay to allow file name display
        setTimeout(processFile, 1000);
    });

    function parseCSV(text) {
        const lines = text.split('\n');
        const headers = lines.shift().split(',');
        return lines.map(line => {
            const data = line.split(',');
            return headers.reduce((obj, nextKey, index) => {
                obj[nextKey] = data[index];
                return obj;
            }, {});
        });
    }

    function hasRequiredColumns(data, requiredColumns) {
        return requiredColumns.every(col => data[0] && data[0].hasOwnProperty(col));
    }

    function processFile() {
        const fileInput = $('#customFile')[0]; // Using jQuery for consistency

        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const reader = new FileReader();

            reader.onload = function(e) {
                const text = e.target.result;
                const data = parseCSV(text);

                if (hasRequiredColumns(data, ['Default'])) {
                    // Actions if required columns are present
                    localStorage.setItem("uploadStatus", "File processed successfully.");
                    localStorage.setItem("columns", JSON.stringify(Object.keys(data[0])));
                    window.location.href = 'result.html'; // Redirect
                } else {
                    // Actions if required columns are missing
                    localStorage.setItem("uploadStatus", "Uploaded file is missing some required columns.");
                    window.location.href = 'result.html'; // Redirect even if there's an error
                }
            };

            reader.readAsText(file);
        } else {
            alert("Please upload a file.");
        }
    }
});
