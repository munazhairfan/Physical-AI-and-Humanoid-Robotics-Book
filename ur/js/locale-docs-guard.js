/**
 * Locale Switching Guard for Docusaurus Docs Pages
 *
 * This script prevents 404 errors when switching locale from docs pages
 * by intercepting the locale switch event and redirecting to appropriate
 * locale-specific documentation paths.
 */

(function() {
  'use strict';

  // Check if we're on a docs page
  function isDocsPage() {
    const pathname = window.location.pathname;
    // Check if path starts with /docs/ or /{locale}/docs/
    return pathname.startsWith('/docs/') ||
           /^\/[a-zA-Z]{2}\/docs\//.test(pathname) ||
           /^\/[a-zA-Z]{2}-[a-zA-Z]{2}\/docs\//.test(pathname);
  }

  // Get the current locale based on URL
  function getCurrentLocale() {
    const pathParts = window.location.pathname.split('/');
    // Check if the first part after domain is a locale code (e.g., 'en', 'es', 'fr', 'ur')
    if (pathParts.length > 1 && pathParts[1] !== 'docs') {
      // It might be a locale prefix like /en/docs/, /es/docs/, /ur/docs/, etc.
      const potentialLocale = pathParts[1];
      // Basic check for locale format (2-3 letters)
      if (/^[a-zA-Z]{2,3}$/.test(potentialLocale)) {
        return potentialLocale;
      }
    }
    return null; // Default locale (no prefix)
  }

  // Generate the target locale-specific docs path
  function getTargetDocsPath(targetLocale) {
    const currentPath = window.location.pathname;
    const currentLocale = getCurrentLocale();

    if (currentLocale) {
      // Current path has locale prefix: /locale/docs/...
      // Remove the current locale prefix and add the target locale
      const pathWithoutLocale = currentPath.substring(currentLocale.length + 1); // +1 for the '/'
      if (targetLocale) {
        return '/' + targetLocale + pathWithoutLocale;
      } else {
        // Target is default locale (no prefix)
        return pathWithoutLocale;
      }
    } else {
      // Current path has no locale prefix (default locale)
      // Add the target locale prefix to the docs path
      if (targetLocale) {
        return '/' + targetLocale + currentPath;
      } else {
        // Already on default locale
        return currentPath;
      }
    }
  }

  // Function to handle locale switching on docs pages
  function handleDocsLocaleSwitch(targetLocale) {
    // Generate the correct docs path for the target locale
    const targetDocsPath = getTargetDocsPath(targetLocale);

    // Redirect to the locale-specific docs path
    window.location.href = targetDocsPath;
  }

  // Listen for clicks on locale switcher elements
  document.addEventListener('click', function(event) {
    // Only handle if we're on a docs page
    if (!isDocsPage()) {
      return;
    }

    // Check if the clicked element is a locale switcher
    let targetElement = event.target;
    while (targetElement && targetElement !== document) {
      const href = targetElement.getAttribute?.('href');
      if (href && (href.startsWith('/en/') || href.startsWith('/es/') || href.startsWith('/fr/') ||
                   href.startsWith('/de/') || href.startsWith('/ja/') || href.startsWith('/ko/') ||
                   href.startsWith('/zh/') || href.startsWith('/ur/'))) {

        // Prevent the default behavior
        event.preventDefault();

        // Extract the target locale from the href
        const localeMatch = href.match(/^\/([a-zA-Z-]{2,5})\//);
        if (localeMatch) {
          const targetLocale = localeMatch[1];
          handleDocsLocaleSwitch(targetLocale);
        }
        break;
      }

      targetElement = targetElement.parentElement;
    }
  }, true);

  // Docusaurus may use history.pushState for client-side navigation
  // Let's intercept that as well
  const originalPushState = history.pushState;
  const originalReplaceState = history.replaceState;

  // Override pushState to detect locale changes
  history.pushState = function() {
    originalPushState.apply(history, arguments);

    // Check if this navigation is happening on a docs page and involves locale change
    const newUrl = arguments[2];
    if (newUrl && typeof newUrl === 'string' && isDocsPage()) {
      const localeMatch = newUrl.match(/^\/([a-zA-Z-]{2,5})\//);
      if (localeMatch) {
        const targetLocale = localeMatch[1];
        // If we're on a docs page and changing locale, ensure proper path
        setTimeout(() => {
          if (isDocsPage()) {
            const correctPath = getTargetDocsPath(targetLocale);
            if (window.location.pathname !== correctPath) {
              window.location.replace(correctPath);
            }
          }
        }, 0);
      }
    }
  };

  history.replaceState = function() {
    originalReplaceState.apply(history, arguments);

    // Similar check for replaceState
    const newUrl = arguments[2];
    if (newUrl && typeof newUrl === 'string' && isDocsPage()) {
      const localeMatch = newUrl.match(/^\/([a-zA-Z-]{2,5})\//);
      if (localeMatch) {
        const targetLocale = localeMatch[1];
        setTimeout(() => {
          if (isDocsPage()) {
            const correctPath = getTargetDocsPath(targetLocale);
            if (window.location.pathname !== correctPath) {
              window.location.replace(correctPath);
            }
          }
        }, 0);
      }
    }
  };

  // Listen for popstate events (browser back/forward buttons)
  window.addEventListener('popstate', function(event) {
    if (isDocsPage()) {
      // After navigation, check if we're on the right locale-specific path
      const currentLocale = getCurrentLocale();
      if (currentLocale) {
        const expectedPath = getTargetDocsPath(currentLocale);
        if (window.location.pathname !== expectedPath && window.location.pathname.startsWith('/docs/')) {
          // Redirect to the proper locale-specific docs path
          window.location.replace(expectedPath);
        }
      }
    }
  });

})();