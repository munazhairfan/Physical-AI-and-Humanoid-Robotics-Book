// Fix locale switching to prevent URL duplication
document.addEventListener('DOMContentLoaded', function() {
  // Intercept clicks on locale dropdown links
  document.addEventListener('click', function(event) {
    const localeLink = event.target.closest('a[href^="/ur/"]');
    if (localeLink) {
      event.preventDefault();

      // Get the target URL
      let targetUrl = localeLink.getAttribute('href');

      // Get current path
      const currentPath = window.location.pathname;

      // If current path already contains /ur/, strip it first
      let cleanPath = currentPath;
      if (currentPath.startsWith('/ur/')) {
        cleanPath = '/' + currentPath.substring(4); // Remove '/ur/'
      }

      // If clean path is just '/' then use empty string
      if (cleanPath === '/') {
        cleanPath = '';
      }

      // Create the proper target URL
      const finalUrl = '/ur' + cleanPath + window.location.search + window.location.hash;

      // Navigate to the correct URL
      window.location.href = finalUrl;
    }
  });
});