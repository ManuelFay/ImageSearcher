(() => {
    const state = {
        query: "",
        results: [],
        selected: null,
        includeFaces: true,
        history: []
    };

    const elements = {
        conversationImages: document.getElementById("conversation-images"),
        detail: document.getElementById("detail-view"),
        faceGroups: document.getElementById("face-groups"),
        faces: document.getElementById("faces-view"),
        form: document.getElementById("search-form"),
        gallery: document.getElementById("gallery"),
        hero: document.getElementById("hero"),
        metadata: document.getElementById("selected-metadata"),
        previousImage: document.getElementById("previous-image"),
        query: document.getElementById("query"),
        results: document.getElementById("results-view"),
        resultsCount: document.getElementById("results-count"),
        resultsTitle: document.getElementById("results-title"),
        returnButton: document.getElementById("return-button"),
        returnInline: document.getElementById("return-inline"),
        searchButton: document.getElementById("search-button"),
        selectedImage: document.getElementById("selected-image"),
        selectedImageButton: document.getElementById("selected-image-button"),
        selectedName: document.getElementById("selected-name"),
        selectedPath: document.getElementById("selected-path"),
        status: document.getElementById("status"),
        template: document.getElementById("card-template"),
        timestampImages: document.getElementById("timestamp-images")
    };

    function mediaUrl(path) {
        return `/media?path=${encodeURIComponent(path)}`;
    }

    function filename(path) {
        return path.split(/[\\/]/).pop() || path;
    }

    function formatScore(score, isDistance = false) {
        if (typeof score !== "number") {
            return "";
        }
        return isDistance ? `distance ${score.toFixed(3)}` : `match ${(score * 100).toFixed(1)}%`;
    }

    function formatDate(value) {
        if (!value) {
            return "Unknown";
        }
        return new Intl.DateTimeFormat(undefined, {
            dateStyle: "medium",
            timeStyle: "short"
        }).format(new Date(value));
    }

    function formatDifference(seconds) {
        if (typeof seconds !== "number") {
            return "";
        }
        const minutes = Math.round(seconds / 60);
        if (minutes < 60) {
            return `${Math.max(minutes, 1)} min apart`;
        }
        const hours = Math.round(minutes / 60);
        if (hours < 48) {
            return `${hours} h apart`;
        }
        return `${Math.round(hours / 24)} days apart`;
    }

    function setStatus(message, isError = false) {
        elements.status.textContent = message;
        elements.status.classList.toggle("hidden", !message);
        elements.status.classList.toggle("error", isError);
    }

    function setBusy(isBusy) {
        elements.searchButton.disabled = isBusy;
        elements.searchButton.textContent = isBusy ? "Searching..." : "Search";
    }

    function makeCard(image, labelMode = "score") {
        const card = elements.template.content.firstElementChild.cloneNode(true);
        const preview = card.querySelector("img");
        preview.src = mediaUrl(image.image_path);
        preview.alt = `Preview of ${filename(image.image_path)}`;
        card.querySelector("strong").textContent = filename(image.image_path);
        let label = formatScore(image.score, labelMode === "distance");
        if (labelMode === "timestamp") {
            label = formatDifference(image.time_difference_seconds);
        }
        card.querySelector("small").textContent = label;
        card.addEventListener("click", () => selectImage(image, true));
        return card;
    }

    function renderResults() {
        elements.gallery.replaceChildren(...state.results.map((image) => makeCard(image)));
        elements.resultsTitle.textContent = `"${state.query}"`;
        const resultWord = state.results.length === 1 ? "result" : "results";
        elements.resultsCount.textContent = `${state.results.length} ${resultWord}`;
        elements.results.classList.remove("hidden");
        elements.detail.classList.add("hidden");
        elements.returnButton.classList.add("hidden");
    }

    function showResults() {
        state.selected = null;
        state.history = [];
        elements.detail.classList.add("hidden");
        elements.results.classList.remove("hidden");
        elements.returnButton.classList.add("hidden");
        setStatus("");
        window.scrollTo({top: 0, behavior: "smooth"});
    }

    function setBackControls() {
        const hasPriorPicture = state.history.length > 1;
        elements.previousImage.classList.toggle("hidden", !hasPriorPicture);
        elements.previousImage.textContent = hasPriorPicture ? "Go back to previous photo" : "Go back";
        elements.returnButton.textContent = hasPriorPicture ? "Back to previous photo" : "Back to results";
    }

    function goBack() {
        if (state.history.length <= 1) {
            showResults();
            return;
        }
        state.history.pop();
        selectImage(state.history[state.history.length - 1], false);
    }

    function addMetadataRow(label, value) {
        if (!value) {
            return;
        }
        const row = document.createElement("div");
        const term = document.createElement("dt");
        const description = document.createElement("dd");
        term.textContent = label;
        description.textContent = value;
        row.append(term, description);
        elements.metadata.append(row);
    }

    function renderMetadata(metadata) {
        elements.metadata.replaceChildren();
        addMetadataRow("Date", formatDate(metadata.captured_at));
        addMetadataRow("Date source", metadata.date_source);
        addMetadataRow("Conversation", metadata.conversation);
        addMetadataRow("Shared by", metadata.sender);
        if (metadata.width && metadata.height) {
            addMetadataRow("Dimensions", `${metadata.width} x ${metadata.height}`);
        }
        if (metadata.size_bytes) {
            addMetadataRow("Size", `${(metadata.size_bytes / 1024 / 1024).toFixed(2)} MB`);
        }
    }

    function renderContextGallery(container, images, emptyMessage) {
        container.replaceChildren();
        if (!images.length) {
            const empty = document.createElement("p");
            empty.className = "face-empty";
            empty.textContent = emptyMessage;
            container.append(empty);
            return;
        }
        container.replaceChildren(...images.map((image) => makeCard(image, "timestamp")));
    }

    function renderFaceGroups(groups) {
        elements.faceGroups.replaceChildren();
        if (!state.includeFaces) {
            elements.faces.classList.add("hidden");
            return;
        }
        elements.faces.classList.remove("hidden");
        if (!groups.length) {
            const empty = document.createElement("p");
            empty.className = "face-empty";
            empty.textContent = "No indexed faces were found for this photo.";
            elements.faceGroups.append(empty);
            return;
        }
        groups.forEach((group, index) => {
            const container = document.createElement("section");
            container.className = "face-group";
            const title = document.createElement("h3");
            title.textContent = `Person ${index + 1}`;
            const gallery = document.createElement("div");
            gallery.className = "gallery";
            gallery.replaceChildren(...group.map((image) => makeCard(image, "distance")));
            container.append(title, gallery);
            elements.faceGroups.append(container);
        });
    }

    async function loadContext(image) {
        try {
            const response = await fetch(`/image_context?path=${encodeURIComponent(image.image_path)}`);
            if (!response.ok) {
                throw new Error("The context request failed.");
            }
            const payload = await response.json();
            if (state.selected === image) {
                renderMetadata(payload.metadata);
                renderContextGallery(elements.conversationImages, payload.same_conversation,
                    "No other photos were found in this conversation.");
                renderContextGallery(elements.timestampImages, payload.nearby_time,
                    "No nearby photos were found.");
            }
        } catch (error) {
            if (state.selected === image) {
                elements.metadata.replaceChildren();
                renderContextGallery(elements.conversationImages, [], "Conversation photos could not be loaded.");
                renderContextGallery(elements.timestampImages, [], "Timestamp matches could not be loaded.");
            }
        }
    }

    async function selectImage(image, addToHistory = true) {
        if (addToHistory) {
            state.history.push(image);
        }
        state.selected = image;
        elements.results.classList.add("hidden");
        elements.detail.classList.remove("hidden");
        elements.returnButton.classList.remove("hidden");
        elements.selectedImage.src = mediaUrl(image.image_path);
        elements.selectedImage.alt = `Selected photo: ${filename(image.image_path)}`;
        elements.selectedName.textContent = filename(image.image_path);
        elements.selectedPath.textContent = image.image_path;
        elements.metadata.replaceChildren();
        renderContextGallery(elements.conversationImages, [], "Loading conversation photos...");
        renderContextGallery(elements.timestampImages, [], "Loading nearby photos...");
        renderFaceGroups([]);
        setBackControls();
        window.scrollTo({top: 0, behavior: "smooth"});
        loadContext(image);

        if (!state.includeFaces) {
            return;
        }
        try {
            const response = await fetch(`/get_closest_faces?q=${encodeURIComponent(image.image_path)}`);
            if (!response.ok) {
                throw new Error("The face match request failed.");
            }
            const payload = await response.json();
            if (state.selected === image) {
                renderFaceGroups(payload.results || []);
            }
        } catch (error) {
            if (state.selected === image) {
                elements.faceGroups.innerHTML = '<p class="face-empty">Face matches could not be loaded.</p>';
            }
        }
    }

    async function search(event) {
        event.preventDefault();
        const query = elements.query.value.trim();
        if (!query) {
            return;
        }
        state.query = query;
        state.selected = null;
        state.history = [];
        setBusy(true);
        setStatus("Searching your indexed photos...");
        elements.hero.classList.add("compact");
        elements.results.classList.add("hidden");
        elements.detail.classList.add("hidden");
        elements.returnButton.classList.add("hidden");
        try {
            const response = await fetch(`/get_best_images?q=${encodeURIComponent(query)}`);
            if (!response.ok) {
                throw new Error("Search request failed.");
            }
            const payload = await response.json();
            state.results = payload.results || [];
            setStatus(state.results.length ? "" : "No photos matched this search. Try a broader description or remove a filter.");
            renderResults();
        } catch (error) {
            state.results = [];
            setStatus("Search could not be completed. Check that the API is running and try again.", true);
        } finally {
            setBusy(false);
        }
    }

    async function loadSettings() {
        try {
            const response = await fetch("/settings");
            if (response.ok) {
                const settings = await response.json();
                state.includeFaces = settings.include_faces !== false;
            }
        } catch (error) {
            state.includeFaces = true;
        }
    }

    elements.form.addEventListener("submit", search);
    elements.returnButton.addEventListener("click", goBack);
    elements.previousImage.addEventListener("click", goBack);
    elements.returnInline.addEventListener("click", showResults);
    elements.selectedImageButton.addEventListener("click", goBack);
    document.querySelectorAll("[data-tag]").forEach((button) => {
        button.addEventListener("click", () => {
            const tag = button.dataset.tag;
            const value = elements.query.value.trim();
            if (!value.includes(tag)) {
                elements.query.value = `${value} ${tag}`.trim();
            }
            elements.query.focus();
        });
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && state.selected) {
            goBack();
        }
    });

    loadSettings();
})();
